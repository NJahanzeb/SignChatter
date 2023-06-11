package com.gdsc.signchatter.NetworkPsl;

import com.google.gson.annotations.SerializedName;

public class PSLPredictionModel {

    @SerializedName("Prediction")
    public String PSLPrediction;

    public PSLPredictionModel(String PSLPrediction) {
        this.PSLPrediction = PSLPrediction;
    }

    public String getPSLPrediction() {
        return PSLPrediction;
    }

    // Setter Methods

    public void setPSLPrediction(String PSLPrediction) {
        this.PSLPrediction = PSLPrediction;
    }
}