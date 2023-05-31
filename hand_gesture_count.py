import cv2
import mediapipe as mp
mphands=mp.solutions.hands
draw=mp.solutions.drawing_utils
hands=mphands.Hands()


cap=cv2.VideoCapture(0)
while True:
    su,image=cap.read()
    image=cv2.flip(image,1)
    imgrgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    res=hands.process(imgrgb)
   # print(res)
    image=cv2.cvtColor(imgrgb,cv2.COLOR_RGB2BGR)
    cv2.rectangle(image,(20,350),(90,440),(0,255,0),cv2.FILLED)
    cv2.rectangle(image,(20,350),(90,440),(255,0,0),4)

    lmlist=[]
    tipids=[4,8,12,16,20] # tips landmark of fingers

    if res.multi_hand_landmarks:
        for handlms in res.multi_hand_landmarks: # recognised hand is saved in multi_hand_landmark attribute
            draw.draw_landmarks(image,handlms,mphands.HAND_CONNECTIONS,draw.DrawingSpec(color=(0,255,0)))
            for id,lm in enumerate(handlms.landmark):
                #print(id,lm)
                cx=lm.x  # collect x coordinate of finger landmarks
                cy=lm.y
                lmlist.append([id,cx,cy])
                #print(lmlist)
    fingerlist=[]   
    if len(lmlist)!=0 and len(lmlist)==21:
        # handling thumb
        if lmlist[12][1]<lmlist[20][1]: # for right hand
            if lmlist[tipids[0]][1]>lmlist[tipids[0]-1][1]: # value of x increses when finger closed
                 fingerlist.append(0) # append 0 when finger closed
            else:
                 fingerlist.append(1) # when finger open 
        else: # left hand
            if lmlist[tipids[0]][1]<lmlist[tipids[0]-1][1]: #  value of x decreses when finger closed
                 fingerlist.append(0)  #append 0 when finger closed
            else:
                 fingerlist.append(1)  # when finger open


        for id in range(1,5):
             if lmlist[tipids[id]][2]>lmlist[tipids[id]-2][2]: # 2 means y axis
                 fingerlist.append(0)
             else:
                 fingerlist.append(1)
    #print(fingerlist)   

    #finger count
        if len(fingerlist)!=0:
            fingercount=fingerlist.count(1)
            print(fingercount)
        cv2.putText(image,str(fingercount),(35,415),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),5)



    cv2.imshow('face_detect',image)
    if cv2.waitKey(1) &0XFF==27:
        break
cap.release()
cv2.destroyAllWindows()

