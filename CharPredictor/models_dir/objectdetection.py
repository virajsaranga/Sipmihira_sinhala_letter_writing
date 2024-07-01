import numpy as np
import cv2
import matplotlib.pyplot as plt

class ObjectDetection:
    
    def __init__(self):
        pass
    
    def showImage(self, image):
        plt.figure(figsize = (10, 10))
        plt.imshow(image)
    
    def check_intersection(self, ground_truth, pred):
        x1 = ground_truth[0] 
        x2 = ground_truth[2]
        y1 = ground_truth[1]
        y2 = ground_truth[3]
        
        s_x1 = pred[0] 
        s_x2 = pred[2]
        s_y1 = pred[1]
        s_y2 = pred[3]
        
        if (x1 <= s_x1 and s_x1 <= x2) and (y1 <= s_y1 and s_y1 <= y2) and \
        (x1 <= s_x2 and s_x2 <= x2) and (y1 <= s_y2 and s_y2 <= y2):
            return True
        else:
            return False
    
    def get_objects(self, img, show=False):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        i = 0
        visited = {}
        for c in contours:
            
            if i==0:
                i+=1
                continue
            
            if (cv2.contourArea(c)) > 10:
                
                x, y, w, h = cv2.boundingRect(c)
                
                crop = img[y:y+h, x:x+w]
                pred = np.array([x, y, x+w, y+h])
                area = w*h
                visited[area] = pred
                
        sorted_visited = dict(sorted(visited.items(), reverse=True))

        final_char = []
        objects = []
        for i in sorted_visited:
            flag = True
            for item in final_char:
                if (self.check_intersection(item, sorted_visited[i])):
                    flag = False
                    break
            
            if flag:
                final_char.append(sorted_visited[i])
                
                x1 = sorted_visited[i][0]
                y1 = sorted_visited[i][1]
                x2 = sorted_visited[i][2]
                y2 = sorted_visited[i][3]
                
                crop = img[y1:y2, x1:x2]
                mask = np.full(img.shape, 255, dtype = np.uint8)
                
                mask_mid_x = mask.shape[1] // 2
                mask_mid_y = mask.shape[0] // 2

                crop_mid_x = crop.shape[1] // 2
                crop_mid_y = crop.shape[0] // 2

                x_offset = mask_mid_x - crop_mid_x
                y_offset = mask_mid_y - crop_mid_y
                mask[y_offset:y_offset+crop.shape[0], x_offset:x_offset+crop.shape[1]] = crop

                if show:
                    self.showImage(mask)
                
                objects.append(mask)
                
        return objects
                
                    