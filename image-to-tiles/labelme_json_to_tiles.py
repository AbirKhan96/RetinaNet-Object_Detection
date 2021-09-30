from os import name, truncate, write
import json
import itertools
from pathlib import Path

import cv2
import labelme

import numpy as np
import matplotlib.pyplot as plt


def is_bbox_inside_bbox(main_bbox, sub_bbox):
    """
    :param main_bbox: big box which may contain sub_bbox 
    :param sub_bbox: sub_bbox which may fall in main_bbox
    """

    sub_bb_top_left, sub_bb_bot_right = sub_bbox
    main_bb_top_left, main_bb_bot_right = main_bbox

    xmin, ymin = main_bb_top_left
    xmax, ymax = main_bb_bot_right

    xmin_, ymin_ = sub_bb_top_left
    xmax_, ymax_ = sub_bb_bot_right

    # is top-left inside
    if ((xmin_ < xmax) and (xmin_ > xmin)) and ((ymin_ < ymax) and (ymin_ > ymax)):
        print('top-left inside')
        # is bottom-right inside
        if ((xmax_ < xmax) and (xmax_ > xmin)) and ((ymax_ < ymax) and (ymax_ > ymax)):
            print('bottom-right inside')
            return True
    return False


def annot_to_tiles(annot, tile_width=512, tile_height=512, visualize_annots=False, only_annotated_tiles=True):
    im_w, im_h = annot['imageWidth'], annot['imageHeight']
    ws = range(0, im_w, tile_width)
    hs = range(0, im_h, tile_height)

    # im = np.ones((im_h, im_w, 3))*0
    b64_data = annot['imageData']
    im = labelme.utils.img_b64_to_arr(b64_data)

    shapes = annot['shapes']
    tiles_imgs = []
    tiles_annots = []
    tiles_fnames = []
    # areas = []

    # xmin, ymin and xmax, ymax belong to tile
    for tile_idx, (xmin, ymin) in enumerate(itertools.product(ws, hs)):
        # temp: skip or perform for specific tiles
        # if tile_idx != 30: 
        #     continue

        xmax = xmin+tile_width
        ymax = ymin+tile_height

        if (xmax>im_w) or (ymax>im_h):
            continue

        # convert to tiles
        tile_top_left = (xmin, ymin)
        tile_bot_right = (xmax, ymax)

        if visualize_annots:
            im = cv2.rectangle(im, tile_top_left, tile_bot_right, (255, 255, 255), 20)

        tile_img = im[ymin:ymax, xmin:xmax]
        # plt.imshow(tile_img)
        # plt.show()
        # break

        tile_shapes = []
        for shape in shapes:
            if shape['shape_type'] != 'rectangle':
                continue

            # get individual annot points in loop
            (r_xmin, r_ymin), (r_xmax, r_ymax) = shape['points']
            r_xmin, r_ymin = int(r_xmin), int(r_ymin)
            r_xmax, r_ymax = int(r_xmax), int(r_ymax)
            
            # area = abs(r_xmax-r_xmin) * abs(r_ymax-r_ymin)
            # areas.append(area)
            
            if visualize_annots:
                im = cv2.rectangle(im, (int(r_xmin), int(r_ymin)), (int(r_xmax), int(r_ymax)), (255, 255, 255), 1)

            # skip if annot is not falling in tile
            top_left_inside = (r_xmin > xmin) and (r_xmin < xmax) and (r_ymin > ymin) and (r_ymin < ymax)
            bot_left_inside = (r_xmax > xmin) and (r_xmax < xmax) and (r_ymax > ymin) and (r_ymax < ymax)
            if not(top_left_inside or bot_left_inside):
                continue

            # this is not correct way to do because the pixel coordinates are
            # have different offsets. not xmin, ymin, xmax, ymax.
            # so, simply subtract with xmin or xmax.
            r_xmin = r_xmin - xmin
            r_xmax = r_xmax - xmin
            r_ymin = r_ymin - ymin
            r_ymax = r_ymax - ymin

            # tile_img = cv2.rectangle(tile_img, (int(r_xmin), int(r_ymin)), (int(r_xmax), int(r_ymax)), (255, 0, 0), 10)
            # plt.imshow(tile_img)
            # plt.show()

            shape['points'] = [[r_xmin, r_ymin], [r_xmax, r_ymax]]
            tile_shapes.append(shape)


        tot_images_with_annots = len(tile_shapes)
        main_image_name = ''.join(annot["imagePath"].split(".")[:-1])
        new_image_name = f'{main_image_name}_part{str(tile_idx).zfill(3)}_{xmin}_{ymin}_{xmax}_{ymax}.JPG'
        fname = new_image_name.split('.')[-2]

        tile_annot = dict(version=annot['version'],
                          flags=annot['flags'],
                          shapes=tile_shapes,
                          imagePath=new_image_name,
                          imageData=labelme.utils.img_arr_to_b64(tile_img).decode(),
                          imageHeight=tile_width,
                          imageWidth=tile_height)

        # each tile is an image and has 
        # annotation associated with it
        if tot_images_with_annots > 0:
            tiles_imgs.append(tile_img)
            tiles_annots.append(tile_annot)
            tiles_fnames.append(fname)

        print('[done]', f'[{tot_images_with_annots}]', fname)

    if visualize_annots:
        plt.imshow(im)
        plt.show()

    return tiles_imgs, tiles_annots, tiles_fnames


def convert(labelme_json_annot_dir, output_dir, visualize_annots=False, write_images=True):
    output_dir.mkdir(exist_ok=True)
    for annot_path in [f for f in labelme_json_annot_dir.iterdir() if ((f.is_file()) and (f.suffix.lower() in ['.json']))]:
        annot = json.load(open(annot_path))
        tiles_ims, tiles_annots, tiles_fnames = annot_to_tiles(annot=annot,
                                                               tile_width=512,
                                                               tile_height=512,
                                                               only_annotated_tiles=True, # make it False to write all tiles. cureently writing only tiles where annotations are present.
                                                               visualize_annots=visualize_annots)

        print('total files to write:', len(tiles_ims))
        for im_arr, im_annot_dic, fname in zip(tiles_ims, tiles_annots, tiles_fnames):
            print('[writing]', fname)
            if write_images:
                path = str(output_dir / (fname+'.jpg'))
                cv2.imwrite(path, im_arr)

            json_path = str(output_dir / (fname+'.json'))
            with open(json_path, 'w') as fp:
                json.dump(im_annot_dic, fp, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    """
    USAGE:
    ---
        LABELME_JSON_ANNOT_DIR: path to dir of json files created by labelme
        OUTPUT_DIR: dir to get json files 
    """

    LABELME_JSON_ANNOT_DIR = Path('/home/itis/Desktop/labelme-annotations-to-tiles/image-to-tiles/labelme_json_files')
    OUTPUT_DIR = Path('/home/itis/Desktop/labelme-annotations-to-tiles')

    convert(LABELME_JSON_ANNOT_DIR, OUTPUT_DIR)
