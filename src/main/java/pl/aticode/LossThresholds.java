package pl.aticode;

import lombok.Getter;

public enum LossThresholds {

    LOSS_THRESHOLD_1(0.025, 1),
    LOSS_THRESHOLD_2(0.017, 2),
    LOSS_THRESHOLD_3(0.012, 3),
    LOSS_THRESHOLD_4(0.01, 4),
    LOSS_THRESHOLD_5(0.008, 5),
    LOSS_THRESHOLD_6(0.0065, 6),
    LOSS_THRESHOLD_7(0.004, 7),
    LOSS_THRESHOLD_8(0.0022, 8),
    LOSS_THRESHOLD_9(0.00185, 9),
    LOSS_THRESHOLD_10(0.00135, 10),
    LOSS_THRESHOLD_11(0.001, 11),
    LOSS_THRESHOLD_12(0.00075, 12),
    LOSS_THRESHOLD_13(0.0005, 13),
    LOSS_THRESHOLD_14(0.00025, 14),
    LOSS_THRESHOLD_15(0.000185, 15),
    LOSS_THRESHOLD_16(0.000135, 16),
    LOSS_THRESHOLD_17(0.0001, 17),
    LOSS_THRESHOLD_18(0.000085, 18),
    LOSS_THRESHOLD_19(0.00007, 19),
    LOSS_THRESHOLD_20(0.00006, 20),
    LOSS_THRESHOLD_21(0.00005, 21),
    LOSS_THRESHOLD_22(0.00004, 22),
    LOSS_THRESHOLD_23(0.000035, 23),
    LOSS_THRESHOLD_24(0.00003, 24),
    LOSS_THRESHOLD_25(0.000025, 25);

    @Getter private final double value;
    @Getter private final int level;

    LossThresholds(double value, int level) {
        this.value = value;
        this.level = level;
    }
}
