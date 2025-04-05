package pl.aticode;

import lombok.Getter;
import lombok.Setter;

@Getter
public class TrainData {

    private double multiplier1;
    private double multiplier2;
    private double actualResult;
    @Setter private double[] predictResultArray;

    public TrainData(double multiplier1, double multiplier2, double actualResult) {
        this.multiplier1 = multiplier1;
        this.multiplier2 = multiplier2;
        this.actualResult = actualResult;
    }

}
