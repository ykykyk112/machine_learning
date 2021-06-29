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

    n_train, n_valid = 50000., 10000.
    
    for i in range(epoch) :
        

        train_loss, valid_loss = 0.0, 0.0
        train_acc, valid_acc = 0.0, 0.0
        boundary_acc, valid_boundary_acc = 0.0, 0.0
        ensemble_acc, valid_ensemble_acc = 0.0, 0.0

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
            sum_loss = (t_loss + b_loss + e_loss)
            sum_loss.backward()
            
            model.optimizer.step()
            #print(idx, '  loss :', t_loss.item())
            _, pred = torch.max(train_output, dim = 1)
            _, boundary_pred = torch.max(boundary_output, dim = 1)
            _, ensemble_pred = torch.max(ensemble_output, dim = 1)

            train_loss += t_loss.item()
            train_acc += torch.sum(pred == train_target.data)
            boundary_acc += torch.sum(boundary_pred == train_target.data)
            ensemble_acc += torch.sum(ensemble_pred == train_target.data)
            

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


                v_loss = model.loss(valid_output, valid_target)
                #print(v_loss.item())
                _, v_pred = torch.max(valid_output, dim = 1)
                _, v_boundary_pred = torch.max(valid_boundary_output, dim = 1)
                _, v_ensemble_pred = torch.max(valid_ensemble_output, dim = 1)

                valid_loss += v_loss.item()
                valid_acc += torch.sum(v_pred == valid_target.data)
                valid_boundary_acc += torch.sum(v_boundary_pred == valid_target.data)
                valid_ensemble_acc += torch.sum(v_ensemble_pred == valid_target.data)


        train_acc = train_acc*(100.)
        valid_acc = valid_acc*(100.)
        boundary_acc = boundary_acc*(100.)
        valid_boundary_acc = valid_boundary_acc*(100.)
        ensemble_acc = ensemble_acc*(100.)
        valid_ensemble_acc = valid_ensemble_acc*(100.)

        curr_lr = model.optimizer.param_groups[0]['lr']
        model.scheduler.step()

        avg_train_loss = train_loss/len(train_loader)
        train_loss_history.append(float(avg_train_loss))

        avg_valid_loss = valid_loss/len(test_loader)
        valid_loss_history.append(float(avg_valid_loss))

        avg_train_acc = train_acc/n_train
        avg_boundary_train_acc = boundary_acc/n_train
        avg_ensemble_train_acc = ensemble_acc/n_train
        train_acc_history.append(float(avg_train_acc))

        avg_valid_acc = valid_acc/n_valid
        avg_boundary_valid_acc = valid_boundary_acc/n_valid
        avg_ensemble_valid_acc = valid_ensemble_acc/n_valid
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
            #print('epoch.{0:3d} \t train_ls : {1:.6f} \t train_ac : {2:.4f}% \t valid_ls : {3:.6f} \t valid_ac : {4:.4f}% \t lr : {5:.5f} \t bdr_train : {6:.4f}% \t bdr_valid : {7:.4f}% \t ens_train : {8:.4f}% \t ens_valid : {9:.4f}% \t alpha : {10:.4f}'.format(i+1, avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc, curr_lr, avg_boundary_train_acc, avg_boundary_valid_acc, avg_ensemble_train_acc, avg_ensemble_valid_acc, float(torch.sigmoid(model.alpha.data))))        
            print('epoch.{0:3d} \t train_ls : {1:.6f} \t train_ac : {2:.4f}% \t valid_ls : {3:.6f} \t valid_ac : {4:.4f}% \t lr : {5:.5f} \t bdr_train : {6:.4f}% \t bdr_valid : {7:.4f}% \t ens_train : {8:.4f}% \t ens_valid : {9:.4f}%'.format(i+1, avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc, curr_lr, avg_boundary_train_acc, avg_boundary_valid_acc, avg_ensemble_train_acc, avg_ensemble_valid_acc))     
        
        if valid_boundary_acc > best_boundary_valid_acc : best_boundary_valid_acc = valid_boundary_acc
        if valid_ensemble_acc > best_ensemble_valid_acc : best_ensemble_valid_acc = valid_ensemble_acc
        if valid_acc > best_valid_acc : best_valid_acc = valid_acc
        if avg_valid_loss < best_valid_loss : 
            best_valid_loss = avg_valid_loss
            #best_loss_parameter = model.state_dict()
            best_epoch = i+1
        
    # np.save('./ImageNet/cam_ret_imagenet_subset_color.npy', best_latest_valid_cam.cpu())
    # #torch.save(best_loss_parameter, './ImageNet/target_imagenet_subset_48.pth')
    #torch.save(best_loss_parameter, './ImageNet/ImageNet_sum_210628/separated_imagenet_inception_noalpha.pth')

    # print('model parameter, grad cam heatmap are saved, best epoch :', best_epoch)
    print('best acc : {0:.4f}%, best boundary acc : {1:.4f}%, best ensemble acc : {2:.4f}%'.format(best_valid_acc/n_valid, best_boundary_valid_acc/n_valid, best_ensemble_valid_acc/n_valid))

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

        avg_train_acc = train_acc/50000.
        train_acc_history.append(float(avg_train_acc))

        avg_valid_acc = valid_acc/10000.
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
        
        if avg_valid_acc > best_valid_acc : best_valid_acc = avg_valid_acc
        if avg_valid_loss < best_valid_loss : 
            best_valid_loss = avg_valid_loss
            #best_loss_parameter = model.state_dict()
            #best_latest_valid_cam = model.latest_valid_cam.detach()
            best_epoch = i+1
        
    #np.save('./ImageNet/cam_ret_imagenet_subset_color.npy', best_latest_valid_cam.cpu())
    #torch.save(best_loss_parameter, './ImageNet/target_imagenet_subset_48.pth')
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
