from src.WheatKernelClassification.pipelines.prediction_pipeline import CustomData,PredictPipeline
from flask import Flask, request, render_template, jsonify
from src.WheatKernelClassification.logger import logging

app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    
    else:
        data = CustomData(

            area= float(request.form.get('area')),
            perimeter= float(request.form.get('perimeter')),
            compactness= float(request.form.get('compactness')),
            length_of_kernel= float(request.form.get('length_of_kernel')),
            width_of_kernel= float(request.form.get('width_of_kernel')),
            asymmetry_coefficient= float(request.form.get('asymmetry_coefficient')),
            length_of_kernel_groove= float(request.form.get('length_of_kernel_groove'))
        )

        final_data = data.get_data_as_dataframe()
        logging.info(f'{final_data}')

        predict_pipeline = PredictPipeline()

        pred = predict_pipeline.predict(final_data)

        variety_labels = {0: 'Canadian', 2: 'Rosa', 1:'Kama'}
        predicted_label = variety_labels[pred[0]]

    
        return jsonify({'result': predicted_label})
    
if __name__ == '__main__':
    app.run()

