from django.shortcuts import render
from .email.email_extractor import extract_eml_content_from_upload
from .model.infer import predict_email

def upload_file(request):

    context = {}

    # if GET call, return upload page
    if request.method == "GET":
        return render(request, 'upload.html', context)

    # if POST call, perform computation and return results page
    if request.method == "POST" and request.FILES.get('file'):
        uploaded_file = request.FILES["file"]
        email_data = extract_eml_content_from_upload(uploaded_file)
        prediction_data = predict_email(email_data)
        context = {
            "display_result": True,
            "from": email_data["sender"],
            "to": email_data["receiver"],
            "datetime": email_data["datetime"],
            "subject": email_data["subject"],
            "body": email_data["body"],
            "meta_model_prediction": prediction_data["meta_model_prediction"],
            "body_model_prediction": prediction_data["body_model_prediction"],
            "combined_prediction": prediction_data["combined_prediction"]
        }
        return render(request, 'upload.html', context)
