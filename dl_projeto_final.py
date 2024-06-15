import cv2
from ultralytics import YOLO

# detecção local usando yolov8n, devido à limitação de tamanho no GitHub o yolov8x usado no Colab não pode ser utilizado.

# carregando arquivo de pesos pré-treinado no Colab
model = YOLO('best.pt')


def detect_objects(frame):
    """Função que faz a detecção em cada frame"""
    
    # converte para o formato de cores do frame de RGB para BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # faz a detecção das dimensões do objeto
    results = model.track(source=frame, conf=0.55, max_det=1)
    # desenha um retângulo em cada frame com base nas dimensões obtidas pelo modelo
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cv2.rectangle(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    # converte de volta para RGB
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    #print(results)
    return frame 

# abre o vídeo de teste
video_path = 'video_teste2.mp4'
cap = cv2.VideoCapture(video_path)

# salva o resultado 
output_path = 'video-resultado-local.mp4'
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# caso não seja criada uma janela, provavelmente haverá erro no Linux caso queira visualizar as detecções sendo feitas
#cv2.namedWindow('frame')

# aplicando frame a frame em um laço
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # faz a detecção
    frame_with_detections = detect_objects(frame)

    # salva cada frame
    out.write(frame_with_detections)

    # visualizar detecções
    #cv2.imshow('frame', frame_with_detections)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break
    
cap.release()
out.release()
cv2.destroyAllWindows()