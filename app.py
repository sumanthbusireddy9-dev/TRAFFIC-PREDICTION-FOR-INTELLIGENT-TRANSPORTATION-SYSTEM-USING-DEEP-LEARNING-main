from re import DEBUG
from flask import Flask,render_template,session,redirect,url_for,request
import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cvlib as cv
from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from cvlib.object_detection import draw_bbox
import operator


app=Flask(__name__)
req_classes = ['bicycle', 'car', 'motorcycle', 'bus', 'truck']
@app.route('/')
def index():
    return render_template("index.html")
vehicles_count={}
vehicles_count_IMAGE={}



@app.route("/upload1", methods=["POST", "GET"])
def upload1():
    global vehicle_countx1
    global output_filenamex1
    if request.method == "POST":
        
        # Retrieve file from the request
        f = request.files['file']
        file_path = os.path.join('static/inputs', f.filename)
        f.save(file_path)

        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

        # Load image
        image = cv2.imread(file_path)
        height, width, channels = image.shape

        # Preprocessing
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Initialization
        class_ids = []
        confidences = []
        boxes = []
        vehicle_countx1 = 0

        # Iterate through detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Threshold for detection
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Max Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes and count vehicles
        for i in indices.flatten():
            box = boxes[i]
            if class_ids[i] in [2, 3, 5, 7]:  # Class IDs for vehicles in COCO dataset
                vehicle_countx1 += 1
                x, y, w, h = box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the output
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print(f"Vehicle Count: {vehicle_countx1}")
        # imgg1=vehicle_count1


        # Save the output image
        output_filenamex1 = f"output_{f.filename}"
        output_path = os.path.join('static/outputs', output_filenamex1)
        cv2.imwrite(output_path, image)

        # return render_template("uploadimages.html", msg=f"Vehicle Count: {vehicle_count1}", image_path=output_path)
        return redirect(url_for('upload2'))
        

    return render_template("uploadimages.html")




@app.route('/upload2',methods=["POST","GET"])
def upload2():
        global vehicle_countx2
        global output_filenamex2
        if request.method=="POST":
            # Retrieve file from the request
            f = request.files['file']
            file_path = os.path.join('static/inputs', f.filename)
            f.save(file_path)

            net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

            # Load image
            image = cv2.imread(file_path)
            height, width, channels = image.shape

            # Preprocessing
            blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Initialization
            class_ids = []
            confidences = []
            boxes = []
            vehicle_countx2 = 0

            # Iterate through detections
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:  # Threshold for detection
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply Non-Max Suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Draw bounding boxes and count vehicles
            for i in indices.flatten():
                box = boxes[i]
                if class_ids[i] in [2, 3, 5, 7]:  # Class IDs for vehicles in COCO dataset
                    vehicle_countx2 += 1
                    x, y, w, h = box
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the output
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            print(f"Vehicle Count: {vehicle_countx2}")


            # Save the output image
            output_filenamex2 = f"output_{f.filename}"
            output_path = os.path.join('static/outputs', output_filenamex2)
            cv2.imwrite(output_path, image)

            # return render_template("uploadimages.html", msg=f"Vehicle Count: {vehicle_count2}", image_path=output_path)
            return redirect(url_for('upload3'))
        return render_template("upload2.html")

@app.route('/upload3',methods=["POST","GET"])
def upload3():
        global vehicle_countx3
        global  output_filenamex3
        if request.method=="POST":
            f = request.files['file']
            file_path = os.path.join('static/inputs', f.filename)
            f.save(file_path)

            net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

            # Load image
            image = cv2.imread(file_path)
            height, width, channels = image.shape

            # Preprocessing
            blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Initialization
            class_ids = []
            confidences = []
            boxes = []
            vehicle_countx3 = 0

            # Iterate through detections
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:  # Threshold for detection
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply Non-Max Suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Draw bounding boxes and count vehicles
            for i in indices.flatten():
                box = boxes[i]
                if class_ids[i] in [2, 3, 5, 7]:  # Class IDs for vehicles in COCO dataset
                    vehicle_countx3 += 1
                    x, y, w, h = box
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the output
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            print(f"Vehicle Count: {vehicle_countx3}")


            # Save the output image
            output_filenamex3 = f"output_{f.filename}"
            output_path = os.path.join('static/outputs', output_filenamex3)
            cv2.imwrite(output_path, image)

            # return render_template("uploadimages.html", msg=f"Vehicle Count: {vehicle_count3}", image_path=output_path)
            return redirect(url_for('upload4'))

        return render_template("upload3.html")

@app.route('/upload4',methods=["POST","GET"])
def upload4():
    global vehicle_countx4
    global output_filenamex4
   
    if request.method=="POST":
        f = request.files['file']
        file_path = os.path.join('static/inputs', f.filename)
        f.save(file_path)

        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

        # Load image
        image = cv2.imread(file_path)
        height, width, channels = image.shape

        # Preprocessing
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Initialization
        class_ids = []
        confidences = []
        boxes = []
        vehicle_countx4 = 0

        # Iterate through detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Threshold for detection
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Max Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes and count vehicles
        for i in indices.flatten():
            box = boxes[i]
            if class_ids[i] in [2, 3, 5, 7]:  # Class IDs for vehicles in COCO dataset
                vehicle_countx4 += 1
                x, y, w, h = box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the output
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print(f"Vehicle Count: {vehicle_countx4}")


        # Save the output image
        output_filenamex4 = f"output_{f.filename}"
        output_path = os.path.join('static/outputs', output_filenamex4)
        cv2.imwrite(output_path, image)

        return render_template("success.html",mh="successfull")
   

    return render_template("upload4.html")


@app.route('/Viewimage1')
def Viewimage1():
    global vehicle_countx1
    global output_filenamex1
    return render_template("Viewimage1.html",image1=output_filenamex1,l1=vehicle_countx1)

@app.route('/Viewimage2')
def Viewimage2():
    global vehicle_countx2
    global output_filenamex2
    # print(fn2,'fn2')
    return render_template("Viewimage2.html",image2=output_filenamex2,l2=vehicle_countx2)

@app.route('/Viewimage3')
def Viewimage3():
    global vehicle_countx3
    global output_filenamex3
    # print(fn3,'fn3')
    return render_template("Viewimage3.html",image3=output_filenamex3,l3=vehicle_countx3)

@app.route('/Viewimage4')
def Viewimage4():
    global vehicle_countx4
    global output_filenamex4
    return render_template("Viewimage4.html",image4=output_filenamex4,l4=vehicle_countx4)


@app.route('/Viewimage11')
def Viewimage11():
    global output_filenamey1, vehicle_county1
    return render_template("Viewimagey1.html",image1=output_filenamey1,l1=vehicle_county1)
@app.route('/Viewimage22')
def Viewimage22():
    # print(fn2,'fn2')
    global vehicle_county2, output_filenamey2
    return render_template("Viewimagey2.html",image2=output_filenamey2,l2=vehicle_county2)

@app.route('/Viewimage32')
def Viewimage32():
    global vehicle_county3, output_filenamey3
    # print(fn3,'fn3')
    return render_template("Viewimagey3.html",image3=output_filenamey3,l3=vehicle_county3)

# @app.route("/Viewprediction")
# def Viewprediction():
#     c=images_name[0]
#     return render_template("Viewprediction.html",img=images_name,c=c,d=max_key)

@app.route('/xjunction')
def xjunction():
    global list1
    global vehicle_countx1, vehicle_countx2, vehicle_countx3, vehicle_countx4
    a,b,c,d=vehicle_countx1,vehicle_countx2,vehicle_countx3,vehicle_countx4
    list1=[a,b,c,d]
    first=[a,c]
    second=[b,d]
    list2 = sorted(list1)

    if a==b==c==d:
        import random
        index = random.randint(0,3)
        return render_template("xjunction.html", val= index)
        
    if list2[-1] - list2[-2] > 10:
        index = list1.index(list2[-1])
        return render_template("xjunction.html", val = index)
    else:
        index1 = list1.index(list2[-1])
        index2 = list1.index(list2[-2])
        return render_template("xjunction.html", val1 = index1, val2 = index2)
    


    
@app.route('/yjunction')
def yjunction():
    global vehicle_county1, vehicle_county2, vehicle_county3
    a,b,c=vehicle_county1,vehicle_county2,vehicle_county3
    li=[a,b,c]
    if max(li)==a:
        return render_template('yjunction.html', a='a')
    if max(li)==b:
        return render_template('yjunction.html', b='b')
    if max(li)==c:
        return render_template('yjunction.html', c='c')

    return render_template('yjunction.html')

@app.route("/uploady1",methods=["POST","GET"])
def uploady1():
    global output_filenamey1, vehicle_county1
    try:
        if request.method=="POST":
            f = request.files['file']
            file_path = os.path.join('static/inputs', f.filename)
            f.save(file_path)

            net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

            # Load image
            image = cv2.imread(file_path)
            height, width, channels = image.shape

            # Preprocessing
            blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Initialization
            class_ids = []
            confidences = []
            boxes = []
            vehicle_county1 = 0

            # Iterate through detections
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:  # Threshold for detection
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply Non-Max Suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Draw bounding boxes and count vehicles
            for i in indices.flatten():
                box = boxes[i]
                if class_ids[i] in [2, 3, 5, 7]:  # Class IDs for vehicles in COCO dataset
                    vehicle_county1 += 1
                    x, y, w, h = box
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the output
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            print(f"Vehicle Count: {vehicle_county1}")


            # Save the output image
            output_filenamey1 = f"outputy1_{f.filename}"
            output_path = os.path.join('static/outputs', output_filenamey1)
            cv2.imwrite(output_path, image)

            return redirect(url_for('uploady2'))
    except:
        return render_template("uploadimagesy1.html",msg="fail")
    return render_template("uploadimagesy1.html")

@app.route('/uploady2',methods=["POST","GET"])
def uploady2():
    global vehicle_county2, output_filenamey2
    try:
        if request.method=="POST":
            f = request.files['file']
            file_path = os.path.join('static/inputs', f.filename)
            f.save(file_path)

            net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

            # Load image
            image = cv2.imread(file_path)
            height, width, channels = image.shape

            # Preprocessing
            blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Initialization
            class_ids = []
            confidences = []
            boxes = []
            vehicle_county2 = 0

            # Iterate through detections
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:  # Threshold for detection
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply Non-Max Suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Draw bounding boxes and count vehicles
            for i in indices.flatten():
                box = boxes[i]
                if class_ids[i] in [2, 3, 5, 7]:  # Class IDs for vehicles in COCO dataset
                    vehicle_county2 += 1
                    x, y, w, h = box
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the output
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            print(f"Vehicle Count: {vehicle_county2}")


            # Save the output image
            output_filenamey2 = f"outputy1_{f.filename}"
            output_path = os.path.join('static/outputs', output_filenamey2)
            cv2.imwrite(output_path, image)
            return redirect(url_for("uploady3"))
    except:
        return render_template("uploady22.html",msg="fail")
    return render_template("uploady22.html")

#
@app.route('/uploady3',methods=["POST","GET"])
def uploady3():
    
    global vehicle_county3, output_filenamey3
    try:
        if request.method=="POST":
            f = request.files['file']
            file_path = os.path.join('static/inputs', f.filename)
            f.save(file_path)

            net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

            # Load image
            image = cv2.imread(file_path)
            height, width, channels = image.shape

            # Preprocessing
            blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Initialization
            class_ids = []
            confidences = []
            boxes = []
            vehicle_county3 = 0

            # Iterate through detections
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:  # Threshold for detection
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply Non-Max Suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Draw bounding boxes and count vehicles
            for i in indices.flatten():
                box = boxes[i]
                if class_ids[i] in [2, 3, 5, 7]:  # Class IDs for vehicles in COCO dataset
                    vehicle_county3 += 1
                    x, y, w, h = box
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the output
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            print(f"Vehicle Count: {vehicle_county3}")


            # Save the output image
            output_filenamey3 = f"outputy1_{f.filename}"
            output_path = os.path.join('static/outputs', output_filenamey3)
            cv2.imwrite(output_path, image)

            return render_template("success1.html", mh="hi")
    except:
        return render_template("upload3y.html",msg="fail")

    return render_template("upload3y.html")



if (__name__)==('__main__'):
    app.run(debug=True)

