package pl.aticode;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class Util {

    public static final double NORMALIZATION_INPUT_RATE = 10.0;
    public static final double NORMALIZATION_OUTPUT_RATE = 100.0;

    private Util() {
    }

    public static double meanSquareLoss(Collection<TrainData> trainDataList) {
        double sumSquare = 0;
        for (TrainData trainData : trainDataList) {
            double error = trainData.getActualResult() - trainData.getPredictResultArray()[0];
            sumSquare += error * error;
        }
        return sumSquare / trainDataList.size();
    }

    public static List<TrainData> getTrainDataList() {
        List<TrainData> trainDataList = new ArrayList<>();
        for (int i = 1; i <= 10; i++) {
            for (int j = 1; j <= 10; j++) {
                trainDataList.add(new TrainData(i / NORMALIZATION_INPUT_RATE, j / NORMALIZATION_INPUT_RATE, i * j / NORMALIZATION_OUTPUT_RATE));
            }
        }
        return trainDataList;
    }

    public static double round(double value, double places) {
        long factor = (long) Math.pow(10, places);
        return (double) Math.round(value * factor) / factor;
    }

}
