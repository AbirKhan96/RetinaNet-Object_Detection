import json

annot_path = '/Users/mac/Desktop/github/projects/igenesys/github/image-to-tiles/labelme_json_files/14front00022_labelme.json'
annot = json.load(open(annot_path))

# print(annot.keys()) 
# ['version', 'flags', 'shapes', 'imagePath', 'imageData', 'imageHeight', 'imageWidth']
im_path = annot['imagePath']
print("========================================")
print('im_path:', im_path) 
print("========================================")

shapes = annot['shapes']
for shape in shapes:
    # print(shape['shape_type']) # 'rectangle' (only)
    # print(shape['points']) # [[x, y], [x, y]
    # print(shape['label']) # 'smalldish'
    # print(shape.keys()) # ['label', 'points', 'group_id', 'shape_type', 'flags']
    break


import labelme

b64_data = annot['imageData']
im = labelme.utils.img_b64_to_arr(b64_data)
print(im.shape)