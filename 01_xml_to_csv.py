import pandas as pd
import xml.etree.ElementTree as xet
from glob import glob

path = glob('./images_labeled/*.xml')

lables_dict = dict(filepath = [], xmin = [], xmax = [], ymin = [], ymax = [])
for filename in path:
    # filename = path[i]
    info = xet.parse(filename)
    root = info.getroot()
    member_object = root.find("object")
    lables_info = member_object.find('bndbox')
    xmin = int(lables_info.find('xmin').text)
    xmax = int(lables_info.find('xmax').text)
    ymin = int(lables_info.find('ymin').text)
    ymax = int(lables_info.find('ymax').text)
    lables_dict['filepath'].append(filename)
    lables_dict['xmin'].append(xmin)
    lables_dict['xmax'].append(xmax)
    lables_dict['ymin'].append(ymin)
    lables_dict['ymax'].append(ymax)


data_frame = pd.DataFrame(lables_dict)
print(data_frame)

data_frame.to_csv('lables.csv', index=False)
# print(xmin, xmax, ymin, ymax)
