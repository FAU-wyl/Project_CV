# run.py
# Project entry script.
# Each Exercise below can be individually uncommented and run.
# Currently, only the final Exercise 8 RAW -> JPG processing pipeline is enabled by default.

from config import PATTERN_CR3, EX1_NPY, EX1_JPG, EX2_RAW, EX5_DIR, EX6_DIR
from test import investigate_bayer_pattern_elegant
import cv2
from pipeline import (
    exercise2_demosaic,
    exercise3_luminosity,
    exercise4_white_balance,
    ex5_show_linear,
    exercise6_initial_hdr,
    ex6_create_hdr,
    exercise7_icam06,
    ex7_apply_icam06,
    process_raw,
    test_four_bayer_patterns,
)


if __name__ == "__main__":

    # ========================================================
    # Exercise 1: Automatically estimate Bayer arrangement for IMG_9939.npy.
    # Use red/green/blue reference areas from the camera JPG.
    # ========================================================
    investigate_bayer_pattern_elegant(
        raw_path=EX1_NPY,
        jpg_path=EX1_JPG
    )
    #
    # ========================================================
    # Exercise 2: Read a single CR3 RAW and perform basic demosaicing.
    # ========================================================
    exercise2_demosaic(
        raw_path=EX2_RAW,
        output_path="exercise2_demosaicing_preview.jpg",
        pattern=PATTERN_CR3
    )
    #
    # ========================================================
    # Exercise 3: Compare different brightness mapping methods
    # (normalized, gamma, log, etc.).
    # ========================================================
    exercise3_luminosity(
        raw_path=EX2_RAW,
        pattern=PATTERN_CR3
    )
    #
    # ========================================================
    # Exercise 4: Add Gray World white balance and save different
    # brightness curve versions.
    # ========================================================
    exercise4_white_balance(
        raw_path=EX2_RAW,
        pattern=PATTERN_CR3
    )

    # ========================================================
    # Exercise 5: Verify sensor linear response using multiple
    # RAW images with different exposures.
    # ========================================================
    ex5_show_linear()

    # ========================================================
    # Exercise 6: Merge HDR RAW according to lecture method
    # and output with log tone mapping.
    # ========================================================
    hdr_float_data = ex6_create_hdr()

    # ========================================================
    # Exercise 7: Use simplified iCAM06 for local tone mapping.
    # ========================================================
    out_fine = ex7_apply_icam06(hdr_float_data, output_range=6.0, d=15, sigma_color=0.8, sigma_space=4.0)
    cv2.imwrite("ex7_icam06_fine.png", out_fine)
    # ========================================================
    # Debug: Output four Bayer phase results to verify correct CR3 arrangement.
    # ========================================================
    # test_four_bayer_patterns(
    #     raw_path=EX2_RAW
    # )

    # ========================================================
    # Exercise 8: Final wrapper function to directly output enhanced JPG from CR3.
    # ========================================================
    process_raw(
        raw_path="exercise_2_data/02/IMG_4782.CR3",
        jpg_path="exercise8_process_raw_result.jpg",
        pattern=PATTERN_CR3
    )
