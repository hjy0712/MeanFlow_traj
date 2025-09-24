import torch
import numpy as np
import cv2
from PIL import Image
from matplotlib import colormaps as cm
from policy_network import NavDP_Policy
from utils.visualization import log_plot_txt
from dataset.dataset3dfront import NavDP_Base_Datset


class NavDP_Agent:
    def __init__(self,
                 image_intrinsic,
                 image_size=224,
                 memory_size=8,
                 predict_size=24,
                 temporal_depth=16,
                 heads=8,
                 token_dim=384,
                 navi_model = "./navdp_test/navdp-cross-modal.ckpt",
                 device='cuda:0'):
        self.image_intrinsic = image_intrinsic
        self.device = device
        self.predict_size = predict_size
        self.image_size = image_size
        self.memory_size = memory_size
        self.navi_former = NavDP_Policy(image_size,memory_size,predict_size,temporal_depth,heads,token_dim,device)
        self.navi_former.load_state_dict(torch.load(navi_model,map_location=self.device),strict=False)
        self.navi_former.to(self.device)
        self.navi_former.eval()
    
    def reset(self,batch_size,threshold):
        self.batch_size = batch_size
        self.stop_threshold = threshold
        self.memory_queue = [[] for i in range(batch_size)]
    def reset_env(self,i):
        self.memory_queue[i] = []
    
    def project_trajectory(self,images,n_trajectories,n_values):
        trajectory_masks = []
        for i in range(images.shape[0]):
            trajectory_mask = np.array(images[i])
            n_trajectory = n_trajectories[i,:,:,0:2]
            n_value = n_values[i]
            for waypoints,value in zip(n_trajectory,n_value):
                norm_value = np.clip(-value*0.1,0,1)
                colormap = cm.get('jet')
                color = np.array(colormap(norm_value)[0:3]) * 255.0
                input_points = np.zeros((waypoints.shape[0],3)) - 0.2
                input_points[:,0:2] = waypoints
                input_points[:,1] = -input_points[:,1]
                camera_z = images[0].shape[0] - 1 - self.image_intrinsic[1][1] * input_points[:,2] / (input_points[:,0] + 1e-8) - self.image_intrinsic[1][2]
                camera_x = self.image_intrinsic[0][0] * input_points[:,1] / (input_points[:,0] + 1e-8) + self.image_intrinsic[0][2]
                for i in range(camera_x.shape[0]-1):
                    try:
                        if camera_x[i] > 0 and camera_z[i] > 0 and camera_x[i+1] > 0 and camera_z[i+1] > 0:
                            trajectory_mask = cv2.line(trajectory_mask,(int(camera_x[i]),int(camera_z[i])),(int(camera_x[i+1]),int(camera_z[i+1])),color.astype(np.uint8).tolist(),5)
                    except:
                        pass
            trajectory_masks.append(trajectory_mask)
        return np.concatenate(trajectory_masks,axis=1)

    def process_image(self,images):
        assert len(images.shape) == 4
        H,W,C = images.shape[1],images.shape[2],images.shape[3]
        prop = self.image_size/max(H,W)
        return_images = []
        for img in images:
            resize_image = cv2.resize(img,(-1,-1),fx=prop,fy=prop)
            pad_width = max((self.image_size - resize_image.shape[1])//2,0)
            pad_height = max((self.image_size - resize_image.shape[0])//2,0)
            pad_image = np.pad(resize_image,((pad_height,pad_height),(pad_width,pad_width),(0,0)),mode='constant',constant_values=0)
            resize_image = cv2.resize(pad_image,(self.image_size,self.image_size))
            resize_image = np.array(resize_image)
            resize_image = resize_image.astype(np.float32) / 255.0
            return_images.append(resize_image)
        return np.array(return_images)

    def process_depth(self,depths):
        assert len(depths.shape) == 4
        depths[depths==np.inf] = 0
        H,W,C = depths.shape[1],depths.shape[2],depths.shape[3]
        prop = self.image_size/max(H,W)
        return_depths = []
        for depth in depths:
            resize_depth = cv2.resize(depth,(-1,-1),fx=prop,fy=prop)
            pad_width = max((self.image_size - resize_depth.shape[1])//2,0)
            pad_height = max((self.image_size - resize_depth.shape[0])//2,0)
            pad_depth = np.pad(resize_depth,((pad_height,pad_height),(pad_width,pad_width)),mode='constant',constant_values=0)
            resize_depth = cv2.resize(pad_depth,(self.image_size,self.image_size))
            resize_depth[resize_depth>5.0] = 0
            resize_depth[resize_depth<0.1] = 0
            return_depths.append(resize_depth[:,:,np.newaxis])
        return np.array(return_depths)
    
    def process_pixel(self,pixel_coords,input_images):
        return_pixels = []
        H,W,C = input_images.shape[1],input_images.shape[2],input_images.shape[3]
        prop = self.image_size/max(H,W)
        for pixel_coord,input_image in zip(pixel_coords,input_images):
            panel_image = np.zeros_like(input_image,dtype=np.uint8)
            min_x = pixel_coord[0] - 10
            min_y = pixel_coord[1] - 10
            max_x = pixel_coord[0] + 10
            max_y = pixel_coord[1] + 10
            
            if min_x <= 0:
                panel_image[:,0:10] = 255
            elif min_y <= 0:
                panel_image[0:10,:] = 255
            elif max_x >= panel_image.shape[1]:
                panel_image[:,panel_image.shape[1]-10:] = 255
            elif max_y >= panel_image.shape[0]:
                panel_image[panel_image.shape[0]-10:,:] = 255
            elif min_x > 0 and min_y > 0 and max_x < panel_image.shape[1] and max_y < panel_image.shape[0]:
                panel_image[min_y:max_y,min_x:max_x] = 255
            
            resize_image = cv2.resize(panel_image,(-1,-1),fx=prop,fy=prop, interpolation=cv2.INTER_NEAREST)
            pad_width = max((self.image_size - resize_image.shape[1])//2,0)
            pad_height = max((self.image_size - resize_image.shape[0])//2,0)
            pad_image = np.pad(resize_image,((pad_height,pad_height),(pad_width,pad_width),(0,0)),mode='constant',constant_values=0)
            resize_image = cv2.resize(pad_image,(self.image_size,self.image_size))
            resize_image = np.array(resize_image)
            resize_image = resize_image.astype(np.float32) / 255.0
            return_pixels.append(resize_image)
        return np.array(return_pixels).mean(axis=-1)
    
    def process_pointgoal(self,goals):
        clip_goals = goals.clip(-10,10)
        clip_goals[:,0] = np.clip(clip_goals[:,0],0,10)
        return clip_goals
    
    def step_nogoal(self,images,depths):
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                input_image = np.pad(input_image,((self.memory_size - input_image.shape[0],0),(0,0),(0,0),(0,0)))
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])    
                input_image = np.array(self.memory_queue[i])
                
            input_images.append(input_image)
        input_image = np.array(input_images)
        input_depth = process_depths
        # cv2.imwrite("input_image.jpg",np.concatenate(self.memory_queue[0],axis=0)*255)
        all_trajectory, all_values, good_trajectory, bad_trajectory = self.navi_former.predict_nogoal_action(input_image,input_depth)
        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0] = good_trajectory[:,:,:,0] * 0.0
            good_trajectory[:,:,:,1] = np.sign(good_trajectory[:,:,:,1].mean())
        trajectory_mask = self.project_trajectory(images,all_trajectory,all_values) 
        return good_trajectory[:,0], all_trajectory, all_values, trajectory_mask
    
    def step_pointgoal(self,goals,images,depths):
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                input_image = np.pad(input_image,((self.memory_size - input_image.shape[0],0),(0,0),(0,0),(0,0)))
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])    
                input_image = np.array(self.memory_queue[i])
                
            input_images.append(input_image)
        input_image = np.array(input_images)
        input_depth = process_depths
        input_goals = self.process_pointgoal(goals)
        # cv2.imwrite("input_image.jpg",np.concatenate(self.memory_queue[0],axis=0)*255)
        all_trajectory, all_values, good_trajectory, bad_trajectory = self.navi_former.predict_pointgoal_action(input_goals,input_image,input_depth)
        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0] = good_trajectory[:,:,:,0] * 0.0
            good_trajectory[:,:,:,1] = np.sign(good_trajectory[:,:,:,1].mean())
        
        print(all_values.max(),all_values.min())
            
        trajectory_mask = self.project_trajectory(images,all_trajectory,all_values) 
        return good_trajectory[:,0], all_trajectory, all_values, trajectory_mask
    
    def step_imagegoal(self,goals,images,depths):
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                input_image = np.pad(input_image,((self.memory_size - input_image.shape[0],0),(0,0),(0,0),(0,0)))
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])    
                input_image = np.array(self.memory_queue[i])
            input_images.append(input_image)
        input_image = np.array(input_images)
        input_depth = process_depths
        input_goals = self.process_image(goals)
        # cv2.imwrite("input_image.jpg",np.concatenate(self.memory_queue[0],axis=0)*255)
        all_trajectory, all_values, good_trajectory, bad_trajectory = self.navi_former.predict_imagegoal_action(input_goals,input_image,input_depth)
        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0] = good_trajectory[:,:,:,0] * 0.0
            good_trajectory[:,:,:,1] = np.sign(good_trajectory[:,:,:,1].mean())
            
        print(all_values.max(),all_values.min())
        trajectory_mask = self.project_trajectory(images,all_trajectory,all_values) 
        return good_trajectory[:,0], all_trajectory, all_values, trajectory_mask

    def step_pixelgoal(self,goals,images,depths):
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                input_image = np.pad(input_image,((self.memory_size - input_image.shape[0],0),(0,0),(0,0),(0,0)))
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])    
                input_image = np.array(self.memory_queue[i])
            input_images.append(input_image)
            
        input_image = np.array(input_images)
        input_depth = process_depths
        input_goals = self.process_pixel(goals,images)
       
        # cv2.imwrite("input_image.jpg",np.concatenate(self.memory_queue[0],axis=0)*255)
        pixel_vis_image = input_image[0,-1].copy() * 255
        pixel_vis_image[np.where(input_goals[0]==1)] = np.array([255,0,0])
        # cv2.imwrite("pixel_goal.jpg",pixel_vis_image)
        all_trajectory, all_values, good_trajectory, bad_trajectory = self.navi_former.predict_pixelgoal_action(input_goals,input_image,input_depth)
        
        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0] = good_trajectory[:,:,:,0] * 0.0
            good_trajectory[:,:,:,1] = np.sign(good_trajectory[:,:,:,1].mean())
        
        trajectory_mask = self.project_trajectory(images,all_trajectory,all_values) 
        return good_trajectory[:,0], all_trajectory, all_values, trajectory_mask
    
    def step_point_image_goal(self,pointgoal,imagegoal,images,depths):
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                input_image = np.pad(input_image,((self.memory_size - input_image.shape[0],0),(0,0),(0,0),(0,0)))
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])    
                input_image = np.array(self.memory_queue[i])
                
            input_images.append(input_image)
        input_image = np.array(input_images)
        input_depth = process_depths
        input_pointgoal = self.process_pointgoal(pointgoal)
        input_imagegoal = self.process_image(imagegoal)
        
        cv2.imwrite("input_image.jpg",np.concatenate(self.memory_queue[0],axis=0)*255)
        all_trajectory, all_values, good_trajectory, bad_trajectory = self.navi_former.predict_ip_action(input_pointgoal,input_imagegoal,input_image,input_depth)
        
        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0] = good_trajectory[:,:,:,0] * 0.0
            good_trajectory[:,:,:,1] = torch.sign(good_trajectory[:,:,:,1].mean())

        import ipdb; ipdb.set_trace()
        
        trajectory_mask = self.project_trajectory(images,all_trajectory,all_values) 
        return good_trajectory[:,0], all_trajectory, all_values, trajectory_mask

if __name__ == "__main__":
    # 从数据集中选取观测和目标点，输出数据集中的轨迹和模型输出的轨迹
    dataset = NavDP_Base_Datset("/mnt/zrh/data/static_nav_from_n1/3dfront_zed",
                                8,24,224,trajectory_data_scale=1.0,scene_data_scale=1.0,preload=False)
    for i in range(10):
        point_goal,image_goal,pixel_goal,memory_images,depth_image,pred_actions,augment_actions,pred_critic,augment_critic,pixel_flag = dataset.__getitem__(i)
        
        # 从数据集中获取内参矩阵
        camera_intrinsic, _, _, _ = dataset.process_data_parquet(i)
        agent = NavDP_Agent(image_intrinsic=camera_intrinsic,device="cuda:0")
        agent.reset(1,0.00000005)

        # 将tensor转换为numpy数组，因为process_image函数需要numpy数组
        memory_images_np = memory_images.numpy()  # (memory_size, image_size, image_size, 3)
        depth_image_np = depth_image.numpy()       # (image_size, image_size, 1)
        point_goal_np = point_goal.numpy()         # (3,)
        
        # 调整维度以匹配step_pointgoal函数的期望格式
        current_image = memory_images_np[-1]  # 记忆序列，但是只取一帧
        current_image_batch = current_image[np.newaxis, ...]  # (1, image_size, image_size, 3)
        # depth_image需要添加batch维度: (1, image_size, image_size, 1)
        depth_image_batch = depth_image_np[np.newaxis, ...]
        # point_goal需要添加batch维度: (1, 3)
        point_goal_batch = point_goal_np[np.newaxis, ...]
        
        # import ipdb; ipdb.set_trace()
        good_trajectory, all_trajectory, all_values, trajectory_mask = agent.step_pointgoal(point_goal_batch, current_image_batch, depth_image_batch)

        # 输出目标点信息，数据集中的监督轨迹信息和预测轨迹信息，将其积分成轨迹点的坐标输出，而不是每个点之间的相对位移
        pred_actions_cumsum = np.cumsum(pred_actions/4.0,axis=0)
        good_trajectory_cumsum = np.cumsum(good_trajectory,axis=0)
        print(f"目标点信息: {point_goal_np}")
        print(f"数据集中的监督轨迹信息: {pred_actions_cumsum}")
        print(f"预测轨迹信息: {good_trajectory_cumsum}")
        print("--------------------------------")