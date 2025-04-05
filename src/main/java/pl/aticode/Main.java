package pl.aticode;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.text.DecimalFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class Main {
    private static final Path path = Path.of("src/main/resources/multiplicationTableResult.csv");
    public static void main(String[] args) {
        int epochs = 5000000;
        double learnFactor = 0.005;
        double momentum = 0.7;
        double lossThreshold = Arrays.stream(LossThresholds.values()).reduce((first, second) -> second).orElseThrow().getValue();
        boolean isShuffle = true;
        try {
            long startTime = System.currentTimeMillis();
            List<TrainData> trainAndTestData = Util.getTrainDataList();
            if (isShuffle) {
                Collections.shuffle(trainAndTestData);
            }
            Set<TrainData> trainDataSet = new HashSet<>(trainAndTestData.subList(0, 64));
            NeuralNetwork neuralNetwork = new NeuralNetwork(epochs, learnFactor, momentum, lossThreshold);
            epochs = neuralNetwork.train(trainDataSet);
            System.out.println("==========================================================");
            for (TrainData trainData : trainDataSet) {
                for (int i = 0; i < trainData.getPredictResultArray().length; i++) {
                    System.out.println("TRAIN DATA - PREDICT of " + trainData.getMultiplier1() * Util.NORMALIZATION_INPUT_RATE + " x " +
                            trainData.getMultiplier2() * Util.NORMALIZATION_INPUT_RATE + " = " + trainData.getPredictResultArray()[i] * Util.NORMALIZATION_OUTPUT_RATE);
                }
            }
            System.out.println("==========================================================");
            long estimatedTime = (System.currentTimeMillis() - startTime) / 1000;
            System.out.println("TRAIN ELAPSED TIME: "+estimatedTime+" sec.");
            System.out.println("==========================================================");
            saveTrainParamToFile(epochs, learnFactor, lossThreshold, estimatedTime, trainDataSet.size(), isShuffle);
            for (TrainData testData : trainAndTestData.subList(65, 99)) {
                double[] inputArray = new double[]{testData.getMultiplier1(), testData.getMultiplier2()};
                for (int i = 0; i < neuralNetwork.getOutputLayer().getNeuronArray().length; i++) {
                    double testResult = neuralNetwork.feedForward(inputArray)[i];
                    System.out.println("TEST DATA - PREDICT of " + testData.getMultiplier1() * Util.NORMALIZATION_INPUT_RATE + " x " +
                            testData.getMultiplier2() * Util.NORMALIZATION_INPUT_RATE + " = " + testResult * Util.NORMALIZATION_OUTPUT_RATE);
                    saveResultToFile(testData, testResult);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private static void saveTrainParamToFile(int epochs, double learnFactor, double lossThreshold, long estimatedTime, int trainSetSize, boolean isShuffle) throws IOException {
        DecimalFormat df = new DecimalFormat();
        df.setMaximumFractionDigits(5);
        String dataToSave = "Test date: "+ LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")) + ",epochs: " + epochs+",learnFactor: " + df.format(learnFactor)
                + ",lossThreshold: " + df.format(lossThreshold) + ",estimatedTime [s]: " + estimatedTime+",trainSetSize: " + trainSetSize+",isShuffle: " + isShuffle+"\n";
        Files.writeString(path, dataToSave, StandardOpenOption.APPEND);
    }


    private static void saveResultToFile(TrainData testResult, double resultPredict) throws IOException {
        DecimalFormat df = new DecimalFormat();
        df.setMaximumFractionDigits(5);
        String dataToSave = "TEST DATA - PREDICT of," + testResult.getMultiplier1() * Util.NORMALIZATION_INPUT_RATE + ",x," + testResult.getMultiplier2() * Util.NORMALIZATION_INPUT_RATE +
                ", =," + df.format(resultPredict * Util.NORMALIZATION_OUTPUT_RATE)+"\n";
        Files.writeString(path, dataToSave, StandardOpenOption.APPEND);
    }
}
