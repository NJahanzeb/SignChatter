package com.gdsc.signchatter.Network;

public class PredictionModel {

    public String Prediction;

    public PredictionModel(String Prediction) {
        this.Prediction = Prediction;
    }

    public String getPrediction() {
        return Prediction;
    }

    // Setter Methods

    public void setPrediction(String Prediction) {
        this.Prediction = Prediction;
    }
}