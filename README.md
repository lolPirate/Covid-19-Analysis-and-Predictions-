# Covid-19 Analysis and Predictions


## Objective
The goal of this project is to predict Covid-19 trends at a acountry level using Deep Learning (LSTM).


## Procedure
- I have used the [PostMan Covid-19 API](https://documenter.getpostman.com/view/10808728/SzS8rjbc?version=latest) for gathering data on a daily basis.
- I conducted analysis on the gathered data using python. Below are a few examples of analysis for India and China.
> **India Analysis** ![Analysis Image India](/plots/analysis/covid-19-analysis-india-latest.jpg)
> **China Analysis** ![Analysis Image China](/plots/analysis/covid-19-analysis-china-latest.jpg)
- I built the LSTM network (currently in prototype) in KERAS.
- Currently the model is trained on data from 6 different countries: India, China, Iran, Australia, Canada and Italy with plans to add more countries in the future.

## Results (Prototype)
The model provides satisfactory predictions for previously unseen data. I tested the model on a few country's 'confirmed' cases. Results can be seen below.

### Germany
#### Analysis 
![Analysis Image Germany](/plots/analysis/covid-19-analysis-germany-latest.jpg)
#### Predictions for confirmed cases
![Prediction Image Germany](/plots/predictions/prototype/covid-19-predictions-germany-latest.jpg)

### Japan
#### Analysis 
![Analysis Image Germany](/plots/analysis/covid-19-analysis-japan-latest.jpg)
#### Predictions for confirmed cases
![Prediction Image Germany](/plots/predictions/prototype/covid-19-predictions-japan-latest.jpg)

## Usage
### Predictions
There are two steps to use the model for predictions:
1. Run the analysis for particular country using the covid_data_analysis.py
    ``````````````````````
    COUNTRY = 'bangladesh'
    ``````````````````````
    Do the necessary changes in this line

2. Comment out the lines as shown in covid_predictor_prototype.py as shown
    ``````````````````````````````````````````````````````````````````````
    if __name__ == '__main__':
        #model = create_model()
        #model = train(model)
        #model.save(os.path.join(MODEL_FOLDER_PATH, MODEL_NAME))
        predict('japan', MODEL_NAME)
    ``````````````````````````````````````````````````````````````````````

### Training
There are two steps to use the model for training 
1. Run the analysis for particular country using the covid_data_analysis.py
    ``````````````````````
    COUNTRY = 'bangladesh'
    ``````````````````````
    Do the necessary changes in this line

2. Un-Comment out the lines as shown in covid_predictor_prototype.py as shown
    ``````````````````````````````````````````````````````````````````````
    if __name__ == '__main__':
        model = create_model()
        model = train(model)
        model.save(os.path.join(MODEL_FOLDER_PATH, MODEL_NAME))
        predict('japan', MODEL_NAME)
    ``````````````````````````````````````````````````````````````````````
    > Note: The trained model will be saved in models folder based on the name provided in the code. The name can be changed by changing the NAME variable in covid_predictor_prototype.py

## Future

I plan to change the model so that it can predict into the future. Currently it just predicts the trend. After that I will create a web application to deploy it.

## Collaboration

If you are interested in collaborating, drop me a mail at [debamalyadawn21@gmail.com](mailto:debamalyadawn21@gmail.com)