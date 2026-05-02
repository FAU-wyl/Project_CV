# main.py

from data_utils import (
    load_example,
    print_data_info,
    get_valid_points_with_indices,
)

from ransac import ransac_plane

from mask_utils import (
    create_mask_from_indices,
    filter_floor_mask,
    get_non_floor_mask,
    get_points_from_mask,
    largest_connected_component,
)

from geometry_utils import (
    estimate_box_dimensions,
    get_box_top_corners_pca,
)

from visualization import (
    show_images,
    show_point_cloud,
    show_mask,
    show_final_result_2d,
    show_final_result_3d,
)


def main():
    """主流程入口：加载数据、检测地面和箱顶、估计尺寸并可视化结果。"""
    # =========================
    # 1. Load data
    # =========================
    example_id = 1

    A, D, PC = load_example(example_id)

    print_data_info(A, D, PC)

    # =========================
    # 2. Visualize original data
    # =========================
    show_images(A, D)
    show_point_cloud(PC, step=8)

    # =========================
    # 3. Prepare valid 3D points
    # =========================
    points, rows, cols = get_valid_points_with_indices(PC)

    print("\nValid points:")
    print("points shape:", points.shape)
    print("rows shape:", rows.shape)
    print("cols shape:", cols.shape)

    # =========================
    # 4. RANSAC: find floor plane
    # =========================
    FLOOR_THRESHOLD = 0.05
    BOX_THRESHOLD = 0.005
    MAX_ITERATIONS = 1000

    floor_model, floor_inliers = ransac_plane(
        points,
        threshold=FLOOR_THRESHOLD,
        max_iterations=MAX_ITERATIONS,
    )

    print("\nFloor plane model:")
    print("normal:", floor_model[0])
    print("d:", floor_model[1])

    # =========================
    # 5. Convert floor inliers to 2D mask
    # =========================
    raw_floor_mask = create_mask_from_indices(
        image_shape=PC.shape[:2],
        rows=rows,
        cols=cols,
        selected_mask=floor_inliers,
    )

    show_mask(raw_floor_mask, "Raw Floor Mask")

    # =========================
    # 6. Filter floor mask
    # =========================
    filtered_floor_mask = filter_floor_mask(
        raw_floor_mask,
        min_floor_size=100,
        max_hole_size=300,
    )

    show_mask(filtered_floor_mask, "Filtered Floor Mask")

    # =========================
    # 7. Get non-floor mask and points
    # =========================
    non_floor_mask = get_non_floor_mask(filtered_floor_mask)

    show_mask(non_floor_mask, "Non-Floor Mask")

    non_floor_points, non_floor_rows, non_floor_cols = get_points_from_mask(
        PC,
        non_floor_mask,
    )

    print("\nNon-floor points:")
    print("points shape:", non_floor_points.shape)
    print("rows shape:", non_floor_rows.shape)
    print("cols shape:", non_floor_cols.shape)

    # =========================
    # 8. RANSAC: find box top plane
    # =========================
    box_model, box_inliers = ransac_plane(
        non_floor_points,
        threshold=BOX_THRESHOLD,
        max_iterations=MAX_ITERATIONS,
    )

    print("\nBox top plane model:")
    print("normal:", box_model[0])
    print("d:", box_model[1])

    # =========================
    # 9. Convert box top inliers to 2D mask
    # =========================
    raw_box_mask = create_mask_from_indices(
        image_shape=PC.shape[:2],
        rows=non_floor_rows,
        cols=non_floor_cols,
        selected_mask=box_inliers,
    )

    show_mask(raw_box_mask, "Raw Box Top Mask")

    # =========================
    # 10. Keep largest connected component
    # =========================
    box_top_mask = largest_connected_component(raw_box_mask)

    show_mask(box_top_mask, "Box Top Largest Component")

    # =========================
    # 11. Estimate box dimensions
    # =========================
    height, length, width = estimate_box_dimensions(
        PC,
        floor_model,
        box_model,
        box_top_mask,
        method="pca",
    )

    print("\nEstimated box dimensions:")
    print(f"Height: {height:.4f} m")
    print(f"Length: {length:.4f} m")
    print(f"Width:  {width:.4f} m")

    corners = get_box_top_corners_pca(PC, box_top_mask)

    print("\nEstimated box top corners:")
    print(corners)

    # =========================
    # 12. Final 2D visualization
    # =========================
    show_final_result_2d(
        filtered_floor_mask,
        box_top_mask,
        title="Visualization of floor, box and box corners",
    )

    # =========================
    # 13. Optional 3D visualization
    # =========================
    show_final_result_3d(
        PC,
        filtered_floor_mask,
        box_top_mask,
        corners=corners,
        step=1,
    )

    # =========================
    # 14. Summary
    # =========================
    print("\nCurrent pipeline finished:")
    print("  Floor plane found.")
    print("  Floor mask filtered.")
    print("  Non-floor points extracted.")
    print("  Box top plane found.")
    print("  Largest box top component extracted.")
    print("  Box dimensions estimated.")
    print("  Final visualizations created.")


if __name__ == "__main__":
    main()