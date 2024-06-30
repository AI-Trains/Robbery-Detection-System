from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,robbery_behavior_detection,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Robbery_Behavior_Type(request):
    if request.method == "POST":

        if request.method == "POST":

            Fid= request.POST.get('Fid')
            event_unique_id= request.POST.get('event_unique_id')
            occurrencedate= request.POST.get('occurrencedate')
            reporteddate= request.POST.get('reporteddate')
            location_type= request.POST.get('location_type')
            premises_type= request.POST.get('premises_type')
            Neighbourhood= request.POST.get('Neighbourhood')
            Longitude= request.POST.get('Longitude')
            Latitude= request.POST.get('Latitude')

        df = pd.read_csv('Datasets.csv')

        def apply_response(Offence_Label):
            if (Offence_Label == 0):
                return 0  # Robbery Mugging
            elif (Offence_Label == 1):
                return 1  # Robbery Purse Snatch

        df['Label'] = df['Offence_Label'].apply(apply_response)

        cv = CountVectorizer()
        X = df['Fid']
        y = df['Label']

        print("RID")
        print(X)
        print("Results")
        print(y)

        cv = CountVectorizer()
        X = cv.fit_transform(X)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Convolution Neural Network-CNN")
        from sklearn.neural_network import MLPClassifier
        mlpc = MLPClassifier().fit(X_train, y_train)
        y_pred = mlpc.predict(X_test)
        testscore_mlpc = accuracy_score(y_test, y_pred)
        accuracy_score(y_test, y_pred)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('MLPClassifier', mlpc))


        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        Fid1 = [Fid]
        vector1 = cv.transform(Fid1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if (prediction == 0):
            val = 'Robbery Mugging'
        elif (prediction == 1):
            val = 'Robbery Purse Snatch'


        print(val)
        print(pred1)

        robbery_behavior_detection.objects.create(
        Fid=Fid,
        event_unique_id=event_unique_id,
        occurrencedate=occurrencedate,
        reporteddate=reporteddate,
        location_type=location_type,
        premises_type=premises_type,
        Neighbourhood=Neighbourhood,
        Longitude=Longitude,
        Latitude=Latitude,
        Prediction=val)

        return render(request, 'RUser/Predict_Robbery_Behavior_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Robbery_Behavior_Type.html')



