import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

def train_eval_model(model, epoch, train_loader, test_loader) :

    train_loss_history = []
    valid_loss_history = []
    train_acc_history = []
    valid_acc_history = []
    best_valid_loss = 0.0
    best_valid_acc = 0.0
    converge_count = 0
    
    for i in range(epoch) :

        train_loss, valid_loss = 0.0, 0.0
        train_acc, valid_acc = 0.0, 0.0

        for train_data, train_target in train_loader :

            model.train()

            model.optimizer.zero_grad()

            train_output = model.forward(train_data)
            t_loss = model.loss(train_output, train_target)
            t_loss.backward()
            model.optimizer.step()

            _, pred = torch.max(train_output, dim = 1)

            train_loss += t_loss.item()
            train_acc += torch.sum(pred == train_target.data)

        with torch.no_grad() :

            for valid_data, valid_target in test_loader :

                model.eval()

                valid_output = model.forward(valid_data)

                v_loss = model.loss(valid_output, valid_target)

                _, v_pred = torch.max(valid_output, dim = 1)

                valid_loss += v_loss.item()
                valid_acc += torch.sum(v_pred == valid_target.data)

        curr_lr = model.optimizer.param_groups[0]['lr']
        model.scheduler.step(float(valid_loss))

        avg_train_loss = train_loss/len(train_loader)
        train_loss_history.append(float(avg_train_loss))

        avg_valid_loss = valid_loss/len(test_loader)
        valid_loss_history.append(float(avg_valid_loss))

        avg_train_acc = train_acc/len(train_loader)
        train_acc_history.append(float(avg_train_acc))

        avg_valid_acc = valid_acc/len(test_loader)
        valid_acc_history.append(float(avg_valid_acc))

        if i%2==0:
            print('epoch.{0:3d} \t train_ls : {1:.6f} \t train_ac : {2:.4f}% \t valid_ls : {3:.6f} \t valid_ac : {4:.4f}% \t lr : {5:.5f}'.format(i+1, avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc, curr_lr))
            
    ret = np.empty((4, len(avg_train_loss)))
    ret[0] = np.asarray(train_loss_history)
    ret[1] = np.asarray(valid_loss_history)
    ret[2] = np.asarray(train_acc_history)
    ret[3] = np.asarray(valid_acc_history)

    if best_valid_acc < avg_valid_acc :
        best_valid_acc = avg_valid_acc

    if (best_valid_loss > avg_valid_loss) or (best_valid_avg - avg_valid_avg) < 0.5 :
        best_valid_loss = avg_valid_loss
        converge_count = 0
    else :
        converge_count += 1
        if converge_count == 7:
            return ret


    return ret


def train_eval_model_gpu(model, epoch, device, train_loader, test_loader, cam_mode, save_path = None) :

    train_loss_history = []
    valid_loss_history = []
    train_acc_history = []
    valid_acc_history = []
    best_valid_loss = 100.0
    best_valid_acc = 0.0
    best_boundary_valid_acc = 0.0
    best_ensemble_valid_acc = 0.0
    converge_count = 0
    best_epoch = 0

    #n_train, n_valid = 1281167., 50000.
    #n_train, n_valid = 71159., 2750.

    subset = False
    
    for i in range(epoch) :
        
        train_loss, valid_loss = 0.0, 0.0
        t5_train_acc, t5_valid_acc = 0.0, 0.0
        train_acc, valid_acc = 0.0, 0.0
        boundary_acc, valid_boundary_acc = 0.0, 0.0
        t5_boundary_acc, t5_valid_boundary_acc = 0.0, 0.0
        ensemble_acc, valid_ensemble_acc = 0.0, 0.0
        t5_ensemble_acc, t5_valid_ensemble_acc = 0.0, 0.0

        model.train()

        for idx, (train_data, train_target) in enumerate(train_loader) :
            if (idx+1)%50==0:
                print(f'{(idx+1)}/{len(train_loader)} is trained\r', end = '')
            train_data, train_target = train_data.to(device), train_target.to(device)

            model.optimizer.zero_grad()
            if cam_mode :
                train_output, _ = model.forward(train_data)
            else :
                #train_output = model(train_data)
                train_output, boundary_output, ensemble_output = model(train_data)
                #train_output = model(train_data, train_target, idx)


            t_loss = model.loss(train_output, train_target)
            #t_loss.backward()
            b_loss = model.boundary_loss(boundary_output, train_target)
            e_loss = model.ensemble_loss(ensemble_output, train_target)
            sum_loss = (t_loss*(0.2) + b_loss*(1.0) + e_loss*(1.0))
            sum_loss.backward()
            
            model.optimizer.step()
            #print(idx, '  loss :', t_loss.item())
            _, pred = torch.max(train_output, dim = 1)
            _, boundary_pred = torch.max(boundary_output, dim = 1)
            _, ensemble_pred = torch.max(ensemble_output, dim = 1)

            batch_size = train_target.size(0)
            
            if not subset :
                _, t5 = train_output.topk(5, 1, True, True)
                t5 = t5.t()
                _, boundary_t5 = boundary_output.topk(5, 1, True, True)
                boundary_t5 = boundary_t5.t()
                _, ensemble_t5 = ensemble_output.topk(5, 1, True, True)
                ensemble_t5 = ensemble_t5.t()


                correct_t5 = (t5 == train_target.unsqueeze(dim=0)).expand_as(t5)
                correct_boundary_t5 = (boundary_t5 == train_target.unsqueeze(dim=0)).expand_as(boundary_t5)
                correct_ensemble_t5 = (ensemble_t5 == train_target.unsqueeze(dim=0)).expand_as(ensemble_t5)

                correct_t5 = correct_t5.reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size)
                correct_boundary_t5 = correct_boundary_t5.reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size)
                correct_ensemble_t5 = correct_ensemble_t5.reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size)

                t5_train_acc += correct_t5.item()
                t5_boundary_acc += correct_boundary_t5.item()
                t5_ensemble_acc += correct_ensemble_t5.item()

            train_loss += t_loss.item()
            train_acc += (torch.sum(pred == train_target.data).item()*(100.0 / batch_size))
            boundary_acc += (torch.sum(boundary_pred == train_target.data).item()*(100.0 / batch_size))
            ensemble_acc += (torch.sum(ensemble_pred == train_target.data).item()*(100.0 / batch_size))
            

        with torch.no_grad():

            model.eval()
            
            for idx, (valid_data, valid_target) in enumerate(test_loader):
                
                valid_data, valid_target = valid_data.to(device), valid_target.to(device)

                model.optimizer.zero_grad()

                if cam_mode :
                    valid_output, _ = model(valid_data)
                else :
                    #valid_output = model(valid_data)
                    valid_output, valid_boundary_output, valid_ensemble_output = model(valid_data)
                    #valid_output = model(valid_data, valid_target, idx, True)

                batch_size = valid_target.size(0)

                v_loss = model.loss(valid_output, valid_target)
                #print(v_loss.item())
                _, v_pred = torch.max(valid_output, dim = 1)
                _, v_boundary_pred = torch.max(valid_boundary_output, dim = 1)
                _, v_ensemble_pred = torch.max(valid_ensemble_output, dim = 1)
                
                if not subset :

                    _, v_t5 = valid_output.topk(5, 1, True, True)
                    v_t5 = v_t5.t()
                    _, v_boundary_t5 = valid_boundary_output.topk(5, 1, True, True)
                    v_boundary_t5 = v_boundary_t5.t()
                    _, v_ensemble_t5 = valid_ensemble_output.topk(5, 1, True, True)
                    v_ensemble_t5 = v_ensemble_t5.t()


                    correct_v_t5 = (v_t5 == valid_target.unsqueeze(dim=0)).expand_as(v_t5)
                    correct_v_boundary_t5 = (v_boundary_t5 == valid_target.unsqueeze(dim=0)).expand_as(v_boundary_t5)
                    correct_v_ensemble_t5 = (v_ensemble_t5 == valid_target.unsqueeze(dim=0)).expand_as(v_ensemble_t5)

                    correct_v_t5 = correct_v_t5.reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size)
                    correct_v_boundary_t5 = correct_v_boundary_t5.reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size)
                    correct_v_ensemble_t5 = correct_v_ensemble_t5.reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size)

                    t5_valid_acc += correct_v_t5.item()
                    t5_valid_boundary_acc += correct_v_boundary_t5.item()
                    t5_valid_ensemble_acc += correct_v_ensemble_t5.item()

                valid_loss += v_loss.item()
                valid_acc += (torch.sum(v_pred == valid_target.data)).item()*(100.0 / batch_size)
                valid_boundary_acc += (torch.sum(v_boundary_pred == valid_target.data)).item()*(100.0 / batch_size)
                valid_ensemble_acc += (torch.sum(v_ensemble_pred == valid_target.data)).item()*(100.0 / batch_size)


        # train_acc = train_acc*(100.)
        # valid_acc = valid_acc*(100.)
        # boundary_acc = boundary_acc*(100.)
        # valid_boundary_acc = valid_boundary_acc*(100.)
        # ensemble_acc = ensemble_acc*(100.)
        # valid_ensemble_acc = valid_ensemble_acc*(100.)

        curr_lr = model.optimizer.param_groups[0]['lr']
        model.scheduler.step()

        avg_train_loss = train_loss/len(train_loader)
        train_loss_history.append(float(avg_train_loss))

        avg_valid_loss = valid_loss/len(test_loader)
        valid_loss_history.append(float(avg_valid_loss))

        avg_train_acc = train_acc/len(train_loader)
        avg_boundary_train_acc = boundary_acc/len(train_loader)
        avg_ensemble_train_acc = ensemble_acc/len(train_loader)
        avg_valid_acc = valid_acc/len(test_loader)
        avg_boundary_valid_acc = valid_boundary_acc/len(test_loader)
        avg_ensemble_valid_acc = valid_ensemble_acc/len(test_loader)
        #train_acc_history.append(float(avg_train_acc))

        if not subset :
            avg_t5_train_acc = t5_train_acc/len(train_loader)
            avg_t5_boundary_acc = t5_boundary_acc/len(train_loader)
            avg_t5_ensemble_acc = t5_ensemble_acc/len(train_loader)
            avg_t5_valid_acc = t5_valid_acc/len(test_loader)
            avg_valid_t5_boundary_acc = t5_valid_boundary_acc/len(test_loader)
            avg_valid_t5_ensemble_acc = t5_valid_ensemble_acc/len(test_loader)
        #valid_acc_history.append(float(avg_valid_acc))

        # Code about early_stopping

        # if (best_valid_loss < avg_valid_loss) or (avg_valid_acc - best_valid_acc) <= 0.3 :
        #     converge_count += 1
        #     if converge_count == 5:
        #         ret = np.empty((4, len(train_loss_history)))
        #         ret[0] = np.asarray(train_loss_history)
        #         ret[1] = np.asarray(valid_loss_history)
        #         ret[2] = np.asarray(train_acc_history)
        #         ret[3] = np.asarray(valid_acc_history)
        #         return ret
        # elif (best_valid_loss > avg_valid_loss) :
        #     best_valid_loss = avg_valid_loss
        #     converge_count = 0

        params = list(model.parameters())
        #first_weight = params[4].item()
        first_weight = 0.
        #second_weight = params[15].item()
        second_weight = 0.

        if i%2==0 or i%2==1:
            if subset :
                print('epoch.{0:3d} \t train_ls : {1:.6f} \t train_ac : {2:.4f}% \t valid_ls : {3:.6f} \t valid_ac : {4:.4f}% \t lr : {5:.5f} \t bdr_train : {6:.4f}% \t bdr_valid : {7:.4f}% \t ens_train : {8:.4f}% \t ens_valid : {9:.4f}%'.format(i+1, avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc, curr_lr, avg_boundary_train_acc, avg_boundary_valid_acc, avg_ensemble_train_acc, avg_ensemble_valid_acc))     
            else :
                print('epoch.{0:3d} \t train_ls : {1:.6f} \t train_ac : {2:.4f}% \t valid_ls : {3:.6f} \t valid_ac : {4:.4f}% \t lr : {5:.5f} \t bdr_train : {6:.4f}% \t bdr_valid : {7:.4f}% \t ens_train : {8:.4f}% \t ens_valid : {9:.4f}%'.format(i+1, avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc, curr_lr, avg_boundary_train_acc, avg_boundary_valid_acc, avg_ensemble_train_acc, avg_ensemble_valid_acc))     
                print('                top-5 acc          \t train_ac : {0:.4f}% \t                    \t valid_ac : {1:.4f}% \t              \t bdr_train : {2:.4f}% \t bdr_valid : {3:.4f}% \t ens_train : {4:.4f}% \t ens_valid : {5:.4f}%'.format(avg_t5_train_acc, avg_t5_valid_acc, avg_t5_boundary_acc, avg_valid_t5_boundary_acc, avg_t5_ensemble_acc, avg_valid_t5_ensemble_acc))
                print(' ')

        if avg_boundary_valid_acc > best_boundary_valid_acc : 
            best_boundary_valid_acc = avg_boundary_valid_acc
            best_boundary_loss_parameter = model.state_dict()
            torch.save(best_boundary_loss_parameter, './ImageNet/ImageNet_Total/best_boundary/separated_boundary_full_imagenet.pth')
            print('boundary parameter saved.')
        if avg_ensemble_valid_acc > best_ensemble_valid_acc : 
            best_ensemble_valid_acc = avg_ensemble_valid_acc
            best_loss_parameter = model.state_dict()
            torch.save(best_loss_parameter, './ImageNet/ImageNet_Total/best_ensemble/separated_ensemble_full_imagenet.pth')
            print('ensemble parameter saved.')
        if avg_valid_acc > best_valid_acc : best_valid_acc = avg_valid_acc
        if avg_valid_loss < best_valid_loss : 
            best_valid_loss = avg_valid_loss
            best_epoch = i+1
        
    # np.save('./ImageNet/cam_ret_imagenet_subset_color.npy', best_latest_valid_cam.cpu())
    # #torch.save(best_loss_parameter, './ImageNet/target_imagenet_subset_48.pth')
        if i % 5 == 0:
           torch.save(model.state_dict(), './ImageNet/ImageNet_Total/separated_boundary_full_imagenet_epoch_{}.pth'.format(i))
           print('every five epoch, parameter is saved.')


    # print('model parameter, grad cam heatmap are saved, best epoch :', best_epoch)
    print('best acc : {0:.4f}%, best boundary acc : {1:.4f}%, best ensemble acc : {2:.4f}%'.format(best_valid_acc, best_boundary_valid_acc, best_ensemble_valid_acc))

    return

    # ret = np.empty((4, len(train_loss_history)))
    # ret[0] = np.asarray(train_loss_history)
    # ret[1] = np.asarray(valid_loss_history)
    # ret[2] = np.asarray(train_acc_history)
    # ret[3] = np.asarray(valid_acc_history)

    # if save_path != None:
    #     model = model.to('cpu')
    #     torch.save(model.state_dict(), save_path)

    # return ret


def train_eval_model_gpu_cam(model, epoch, device, train_loader, test_loader, cam_mode, save_path = None) :

    train_loss_history = []
    valid_loss_history = []
    train_acc_history = []
    valid_acc_history = []
    best_valid_loss = 100.0
    best_valid_acc = 0.0
    converge_count = 0
    best_epoch = 0

    
    for i in range(epoch) :
        

        train_loss, valid_loss = 0.0, 0.0
        train_acc, valid_acc = 0.0, 0.0

        model.train()

        for idx, (train_data, train_target) in enumerate(train_loader) :
            if (idx+1)%50==0:
                print(f'{(idx+1)}/{len(train_loader)} is trained\r', end = '')
            train_data, train_target = train_data.to(device), train_target.to(device)

            model.optimizer.zero_grad()
            if cam_mode :
                train_output, _ = model.forward(train_data)
            else :
                train_output = model(train_data)
                #train_output = model(train_data, train_target, idx)

            t_loss = model.loss(train_output, train_target)
            t_loss.backward()


            model.optimizer.step()
            #print(idx, '  loss :', t_loss.item())
            _, pred = torch.max(train_output, dim = 1)

            train_loss += t_loss.item()
            train_acc += torch.sum(pred == train_target.data)
            

        with torch.no_grad():

            model.eval()
            
            for idx, (valid_data, valid_target) in enumerate(test_loader):
                
                valid_data, valid_target = valid_data.to(device), valid_target.to(device)

                model.optimizer.zero_grad()

                if cam_mode :
                    valid_output, _ = model(valid_data)
                else :
                    valid_output = model(valid_data)
                    #valid_output = model(valid_data, valid_target, idx, True)


                v_loss = model.loss(valid_output, valid_target)
                #print(v_loss.item())
                _, v_pred = torch.max(valid_output, dim = 1)

                valid_loss += v_loss.item()
                valid_acc += torch.sum(v_pred == valid_target.data)


        train_acc = train_acc*(100.)
        valid_acc = valid_acc*(100.)

        curr_lr = model.optimizer.param_groups[0]['lr']
        model.scheduler.step()

        avg_train_loss = train_loss/len(train_loader)
        train_loss_history.append(float(avg_train_loss))

        avg_valid_loss = valid_loss/len(test_loader)
        valid_loss_history.append(float(avg_valid_loss))

        avg_train_acc = train_acc/71159.
        train_acc_history.append(float(avg_train_acc))

        avg_valid_acc = valid_acc/2750.
        valid_acc_history.append(float(avg_valid_acc))

        # Code about early_stopping

        # if (best_valid_loss < avg_valid_loss) or (avg_valid_acc - best_valid_acc) <= 0.3 :
        #     converge_count += 1
        #     if converge_count == 5:
        #         ret = np.empty((4, len(train_loss_history)))
        #         ret[0] = np.asarray(train_loss_history)
        #         ret[1] = np.asarray(valid_loss_history)
        #         ret[2] = np.asarray(train_acc_history)
        #         ret[3] = np.asarray(valid_acc_history)
        #         return ret
        # elif (best_valid_loss > avg_valid_loss) :
        #     best_valid_loss = avg_valid_loss
        #     converge_count = 0

        params = list(model.parameters())
        #first_weight = params[4].item()
        first_weight = 0.
        #second_weight = params[15].item()
        second_weight = 0.

        if i%2==0 or i%2==1:
            print('epoch.{0:3d} \t train_ls : {1:.6f} \t train_ac : {2:.4f}% \t valid_ls : {3:.6f} \t valid_ac : {4:.4f}% \t lr : {5:.5f} \t bdr_train : {6:.4f}% \t bdr_valid : {7:.4f}%'.format(i+1, avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc, curr_lr, first_weight, second_weight))        
        
        if avg_valid_acc > best_valid_acc : 
            best_valid_acc = avg_valid_acc
            best_acc_parameter = model.state_dict()
            best_epoch = i+1
        if avg_valid_loss < best_valid_loss : 
            best_valid_loss = avg_valid_loss
            #best_loss_parameter = model.state_dict()
            #best_latest_valid_cam = model.latest_valid_cam.detach()
        
    #np.save('./ImageNet/cam_ret_imagenet_subset_2_0630.npy', best_latest_valid_cam.cpu())
    #torch.save(best_acc_parameter, './ImageNet/ImageNet_sum_210701/baseline_best_accuracy_vgg19_reproduce.pth')
    #torch.save(best_loss_parameter, './ImageNet/target_imagenet_subset_48.pth')

    print('model parameter, grad cam heatmap are saved, best epoch :', best_epoch)
    print('best loss : {0:.6f}, base acc : {1:.4f}'.format(best_valid_loss, best_valid_acc))

    return

    # ret = np.empty((4, len(train_loss_history)))
    # ret[0] = np.asarray(train_loss_history)
    # ret[1] = np.asarray(valid_loss_history)
    # ret[2] = np.asarray(train_acc_history)
    # ret[3] = np.asarray(valid_acc_history)

    # if save_path != None:
    #     model = model.to('cpu')
    #     torch.save(model.state_dict(), save_path)

    # return ret



def save_model(model, epoch, train_loader, test_loader, path, cam_mode = False) :

    is_cuda = torch.cuda.is_available()
    if is_cuda :
        print('GPU is available')
        device = torch.device('cuda')
        model = model.to(device)
        history = train_eval_model_gpu(model, epoch, device, train_loader, test_loader, cam_mode)
    else :
        history = train_eval_model(model, epoch, train_loader, test_loader)

    torch.save(model.state_dict(), path)
    print('save complete. saving path : {}'.format(path))
    return history

def save_plot(history, save_path, show = False) :
    
    fig = plt.figure(figsize = (15, 6))

    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.plot(history[0], label = "training loss")
    ax1.plot(history[1], label = "validation loss")
    ax1.legend()
    ax1.set_title('Loss')
    
    ax2.plot(history[2], label = "training accuracy")
    ax2.plot(history[3], label = "validation accuracy")
    ax2.axvline(np.where(history[3]==np.max(history[3]))[0][0], color = 'b', linestyle = '--', linewidth = 2)
    ax2.text(np.where(history[3]==np.max(history[3]))[0][0]+3, np.max(history[3]), '{:.2f}'.format(np.max(history[3]))+'%', color='k', fontsize = 'large', horizontalalignment = 'left',verticalalignment='bottom')
    ax2.legend()
    ax2.set_title('Score')
    
    plt.savefig(save_path)

    if show :
        plt.show()
