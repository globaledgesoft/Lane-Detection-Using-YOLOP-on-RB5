import cv2
import numpy as np
import argparse
import qcsnpe as qc

CPU = 0
GPU = 1
DSP = 2

IMG_WIDTH = 640
IMG_HEIGHT = 640
video_path = "test.mp4"
model_path = "model_data/lanedetection_640x640.dlc";
print(model_path)
out_layers = np.array(["Concat_1387", "Sigmoid_1512", "Sigmoid_1637"])

model = qc.qcsnpe(model_path, out_layers, CPU)

def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2] 
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) 
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h 

    dw = pad_w // 2
    dh = pad_h // 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img
    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  

def inference_yolop(vid):
    if vid == None :
        print("required command line args atleast ----img_folder <image folder path> or --vid <cam/video_path>")
        exit(0)
    if vid is not None:
        if vid == "cam":
            cap = cv2.VideoCapture("tcp://localhost:8080") #RB5 Gstreamer input
        else:
            cap = cv2.VideoCapture(vid)

    start_time = 0 
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time*30)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('output.mp4', fourcc, 10.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        height, width, _ = frame.shape

        # convert to RGB
        img_rgb = frame[:, :, ::-1].copy()

        # pre-processing
        canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
        img = canvas.copy()

        output = model.predict(img)
        #post-processing
        ll_seg_out = output["lane_line_seg"]
        ll_seg_out = np.reshape(ll_seg_out,(1,IMG_HEIGHT,IMG_WIDTH,2))
        ll_seg_out = np.transpose(ll_seg_out, axes=(0,3,1,2))
        ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
        ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)
        color_area = np.zeros((new_unpad_h, new_unpad_w, 3), dtype=np.uint8)
        color_area[ll_seg_mask == 1] = [0, 255, 0]
        color_seg = color_area

        # convert to BGR
        color_seg = color_seg[..., ::-1]
        color_mask = np.mean(color_seg, 2)
        img_merge = canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :]
        img_merge = img_merge[:, :, ::-1]

        img_merge[color_mask != 0] = \
            img_merge[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
        img_merge = img_merge.astype(np.uint8)
        img_merge = cv2.resize(img_merge, (width, height),
                            interpolation=cv2.INTER_LINEAR)


        cv2.imwrite("out.jpg", img_merge)
        frame_c = cv2.resize(img_merge,(640, 480))
        out_video.write(frame_c)
        cv2.imshow("out", frame_c)
        cv2.waitKey(25)



    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--vid", default=None,help="cam/video_path")
    args = vars(ap.parse_args())
    vid = args["vid"]
	
    inference_yolop(vid)
    
