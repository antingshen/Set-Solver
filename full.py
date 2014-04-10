import cv2
import numpy as np
import os, os.path as osp
import sys
import argparse
import logging
import random

log_format = '%(created)f:%(levelname)s:%(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)  # log to file filename='example.log',
TAG = "set-full:"

def full_path(rel_path):
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    return os.path.join(__location__, rel_path)

def rectify(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

def find_cards(img, threshold, wh_ratio=0.3, area_ratio=0.7):
    height, width, depth = img.shape
    timg = img.copy()
    pyr = np.zeros((height/2, width/2, depth), np.uint8)

    cv2.pyrDown(timg, pyr)
    cv2.pyrUp(pyr, timg)

    card_boundaries = []
    cards = []

    gray = cv2.cvtColor(timg, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 0, threshold, apertureSize=5)
    contours, h = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = cv2.arcLength(contour, True)*0.02
        poly = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(poly) != 4:
            continue
        area = cv2.contourArea(poly)
        if area < 30:
        	continue
        rect = cv2.minAreaRect(poly)
        width, height = rect[1]
        if width/height < wh_ratio or height/width < wh_ratio:
            continue
        rect_area = width*height
        if area/rect_area > area_ratio:
            card_boundaries.append(poly)

    for i, card in enumerate(card_boundaries):
        perimeter = cv2.arcLength(card, True)
        approx = rectify(cv2.approxPolyDP(card, 0.02*perimeter, True))
        h = np.array([[0,0],[199,0],[199,299],[0,299]],np.float32)
        transform = cv2.getPerspectiveTransform(approx, h)
        warp = cv2.warpPerspective(timg, transform, (200,300))
        cards.append(warp)

    return card_boundaries, cards


class Card:
    attrs = ('color', 'count', 'shade', 'shape')


    @classmethod
    def instantiate_training_cards(cls, path):
        directory = osp.join(osp.dirname(os.path.realpath(__file__)), path)
        cls.training = []
        for filename in os.listdir(directory):
            card = Card()
            card.unkey(filename.split('.')[0])
            card.image = cv2.imread(osp.join(directory, filename))
            card.preprocess()
            cls.training.append(card)

    def __init__(self, data=None):
        if data is None:
            return
        if type(data) == dict:
            self.__dict__.update(data)
            return
        if type(data) == str:
            self.unkey(data)
            return
        self.image = data
        self.preprocess()
        self.match()

    def preprocess(self):
        self.raw_image = self.image

        self.image = self.image[5:-5,5:-5]
        channels = cv2.split(self.image)
        self.bg_colors = tuple(map(lambda x:np.average(np.hstack((x[:5], x[-5:]))), channels))
        self.image = self.image[5:-5,5:-5]

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 0, 350, apertureSize=5)
        self.canny = edges
        contours, h = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        rect = cv2.boundingRect(np.vstack(contours))
        self.rect = rect
        rect_height = rect[3]
        count = rect_height/100 + 1
        self.count = str(count)

        self.image = self.image[rect[1]:rect[1]+70, 13:-13]

        channels = cv2.split(self.image)
        fg_colors = tuple(map(np.average, channels))
        self.color_diff = tuple(map(lambda x,y:x-y, fg_colors, self.bg_colors))

        if max(self.color_diff) == self.color_diff[1]:
            self.color = 'g'
        elif max(self.color_diff) == self.color_diff[2]:
            self.color = 'r'
        else:
            self.color = 'p'

    def distance(self, training):
        img1 = self.image
        img2 = training.image
        # img1 = cv2.GaussianBlur(img1,(5,5),5)
        # img2 = cv2.GaussianBlur(img2,(5,5),5)    
        diff = cv2.absdiff(img1,img2)
        # diff = cv2.GaussianBlur(diff,(5,5),5)    
        flag, diff = cv2.threshold(diff, 64, 255, cv2.THRESH_BINARY) 
        ret = np.sum(diff)
        return ret

    def match(self):

        closest = min(Card.training, key=self.distance)
        self.closest_distance = self.distance(closest)
        self.shape = closest.shape
        self.shade = closest.shade
        # channels = cv2.split(self.image)
        # self.image = cv2.merge(map(cv2.equalizeHist, channels))


    @property
    def key(self):
        ret = getattr(self, '_key', None)
        if ret:
            return ret
        ret = "_".join((getattr(self, attr, 'null') for attr in Card.attrs))
        self._key = ret
        return ret

    def unkey(self, key):
        for val, attr in zip(key.split('_'), Card.attrs):
            setattr(self, attr, val)

    def __eq__(self, other):
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        return self.key


def recognize_cards(card_boundaries, card_images):
    cards = set()
    for image, boundary in zip(card_images, card_boundaries):
        card = Card(image)
        card.boundary = boundary
        cards.add(card)
    return list(cards)

def find_set(cards):
    curr_set = []
    for num1 in range(len(cards)-2):
        for num2 in range(1,len(cards)-1):
            for num3 in range(2,len(cards)):
                curr_set = [cards[num1], cards[num2], cards[num3]]
                color = len(set([curr_set[0].color, curr_set[1].color, curr_set[2].color]))
                count = len(set([curr_set[0].count, curr_set[1].count, curr_set[2].count]))
                shade = len(set([curr_set[0].shade, curr_set[1].shade, curr_set[2].shade]))
                shape = len(set([curr_set[0].shape, curr_set[1].shape, curr_set[2].shape]))
                if (2 not in [color, count, shade, shape]) and (3 in [color, count, shade, shape]):
                    return curr_set
    return []

def main():
    logging.debug(TAG + "inside main")
    rand = random.Random()

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", help="capture from video file instead of from camera")
    parser.add_argument("-i", "--image", help="use image instead of video")
    parser.add_argument("-e", "--extract", help="extracts cards from image for manual labeling")
    parser.add_argument("-t", "--testing", help="displays one parsed card and its guess")
    args = parser.parse_args()

    Card.instantiate_training_cards('training')

    logging.debug(TAG + "done parsing arguments")

    if args.extract:
        args.image = args.extract
    if args.image:
        oriimage = cv2.imread(args.image)
        newx,newy = oriimage.shape[1]/4,oriimage.shape[0]/4 #new size (w,h)
        frame = cv2.resize(oriimage,(newx,newy))
        image_only = True

    else:
        image_only = False
        capture = cv2.VideoCapture()
        if args.video:
            capture.open(args.video)
        else:
            capture.open(0)
        if not capture.isOpened():
            # Failed to open camera
            return False
        logging.debug(TAG + "camera opened")

    while True:
        if not image_only:
            # logging.debug(TAG + "before reading frame")
            retval, frame = capture.read()
            if not retval:
                break  # end of video
            # logging.debug(TAG + "after reading frame")

        img = frame.copy()

        threshold = 800
        card_boundaries, card_images = find_cards(img, threshold)
        # logging.debug("%s cards found" % len(card_boundaries))

        if args.extract:
            for image_only in card_images:
                filename = '  %020x.png' % random.randrange(16**20)
                cv2.imwrite(filename, image_only)
            break

        cards = recognize_cards(card_boundaries, card_images)

        if args.testing:
            card = rand.choice(cards)
            cv2.destroyAllWindows()
            cv2.imshow(card.key, card.raw_image)
            print card.color_diff
            key = cv2.waitKey(0)
            if key == 27 or key == 1048603:
                break
            continue

        sets = find_set(cards)
        bounds = [card.boundary for card in sets]

        height = 720
        width = 1280
        blank = np.ones((height, width, 3), np.uint8)
        blank[:, 0:width] = (255, 255, 255)  # convert to white

        cv2.drawContours(img, bounds, -1, (255, 0, 0), 3)
        cv2.imshow("all_cards", img)

        if image_only:
            cv2.waitKey(0)
            break
        logging.debug(sets)
        if cv2.waitKey(5) == 27:  # exit on escape
            logging.debug(TAG + "received escape key")
            break

    return True

logging.debug(TAG + "starting module")
if __name__ == "__main__":
    logging.debug(TAG + "starting main")
    main()
