import easyocr
import cv2
import layoutparser as lp
import layoutparser.visualization as lp_viz
from paddleocr import PaddleOCR


NN_CONFIG_AND_MAP = [
  'lp://NewspaperNavigator/faster_rcnn_R_50_FPN_3x/config',
  {
    0: 'Photograph',
    1: 'Illustration',
    2: 'Map',
    3: 'Comics/Cartoon',
    4: 'Editorial Cartoon',
    5: 'Headline',
    6: 'Advertisement',
  },
]

HJD_CONFIG_AND_MAP = [
  'lp://NewspaperNavigator/faster_rcnn_R_50_FPN_3x/config',
  {
    1: 'Page Frame',
    2: 'Row',
    3: 'Title Region',
    4: 'Text Region',
    5: 'Title',
    6: 'Subtitle',
    7: 'Other',
  },
]

ocr = PaddleOCR(
    use_doc_orientation_classify=False, 
    use_doc_unwarping=False, 
    use_textline_orientation=False)
    # enable_mkldnn=False)


def generate_hocr(layout, image, reader):
  pass


def process(img_path, vizualize, model_choice, gpu):
  # image = cv2.imread(str(img_path))
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # match model_choice:
  #   case 'nn':
  #     model_config, label_map = NN_CONFIG_AND_MAP
  #   case 'hjd':
  #     model_config, label_map = HJD_CONFIG_AND_MAP
  
  result = ocr.predict(str(img_path))

  # reader = easyocr.Reader(['en'], gpu=gpu)
  # model = lp.Detectron2LayoutModel(
  #   model_config,
  #   extra_config=['MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.5],
  #   label_map=label_map,
  # )

  # result = reader.readtext(str(img_path))
  # layout = model.detect(image)

  if vizualize:
    # lp_viz.draw
    # viz = lp.draw_layout(image, layout)
    # viz.save(img_path)

    # image = cv2.imread(str(img_path))
    # for _,  (bbox, text, conf) in enumerate(result):
    #   x_coords = [p[0] for p in bbox]
    #   y_coords = [p[1] for p in bbox]
    #   x1, y1, x2, y2 = int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))
    #   cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imwrite(str(img_path), image)

    for res in result:
      # res.print()
      res.save_to_img(img_path)
      # res.save_to_json("output")




