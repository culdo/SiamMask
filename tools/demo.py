# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from threading import Thread

from tools.test import *

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--video', default='', required=True, help='datasets')
parser.add_argument('--start-frame', default=0, type=int, help='start frame')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

if args.video.startswith("/") or args.video.startswith("."):
    is_file = True
else:
    is_file = False

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# Setup Model
cfg = load_config(args)
from custom import Custom

siammask = Custom(anchors=cfg['anchors'])
if args.resume:
    assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
    siammask = load_pretrain(siammask, args.resume)

siammask.eval().to(device)

# Parse Image file
if args.video.isnumeric():
    cap = cv2.VideoCapture(int(args.video))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
else:
    cap = cv2.VideoCapture(args.video)
cap.set(1, args.start_frame)

# Select ROI
cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
paused = is_file
while cap.isOpened():
    key = cv2.waitKey(1)
    if key == ord(" "):
        paused = not paused
    elif key > 0:
        break
    if not paused or cap.get(cv2.CAP_PROP_POS_FRAMES) == args.start_frame:
        ret, frame = cap.read()
        cv2.imshow('SiamMask', frame)


def _init():
    try:
        x, y, w, h = cv2.selectROI('SiamMask', frame, False, False)
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        state = siamese_init(frame, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        return state
    except Exception as e:
        print(e)
        exit()


state = _init()

key = 0

refPt = []
cropping = False


def click_and_crop(event, x, y):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", frame)


def capture_thread():
    def fun():
        global frame
        global key
        while cap.isOpened():
            ret, frame = cap.read()
            if key > 0:
                break

    Thread(target=fun).start()


if __name__ == '__main__':

    toc = 0
    i = 0
    if not is_file:
        capture_thread()
    while cap.isOpened():
        if is_file:
            ret, frame = cap.read()
        if frame is None:
            break
        tic = cv2.getTickCount()
        # tracking
        result = frame.copy()
        state = siamese_track(state, result, mask_enable=True, refine_enable=True, device=device)  # track
        location = state['ploygon'].flatten()
        mask = state['mask'] > state['p'].seg_thr

        result[:, :, 2] = (mask > 0) * 255 + (mask == 0) * result[:, :, 2]
        cv2.rectangle(result, cv2.boundingRect(np.uint8(mask)), (255, 0, 0))
        cv2.imshow('SiamMask', result)
        key = cv2.waitKey(1)
        if key > 0:
            break
        i += 1

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = i / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
