from __future__ import annotations

import numpy as np
import streamlit as st

from geometry import compute_distortion, compute_mapping, summarize_mapping
from imaging import (
    array_to_pil,
    build_default_input,
    compose_image,
    draw_bbox_overlay,
    fit_image_to_mask,
    forward_simulation,
    image_to_bytes,
    inverse_design,
    make_heatmap,
    overlay_grid,
    paper_usage_bbox,
    tint_invalid_regions,
)
from metrics import distortion_stats, mse, psnr, ssim_approx
from utils import DEFAULTS, format_float, format_ratio, open_uploaded_image, params_from_dict


st.set_page_config(page_title="Reflect Art App", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e7ecf5;
        padding: 0.7rem 0.8rem;
        border-radius: 14px;
    }
    .small-note {color: #546173; font-size: 0.93rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Reflect Art App")
st.caption("用于数学建模展示的圆柱镜反射成像交互原型")


def sidebar_controls():
    st.sidebar.header("模式")
    mode = st.sidebar.radio("运行模式", ["正向模拟", "逆向设计", "双向验证"], index=2)
    if st.sidebar.button("恢复默认参数", use_container_width=True):
        for key, value in DEFAULTS.items():
            st.session_state[key] = value
        st.rerun()

    st.sidebar.header("几何参数")
    config = {}
    config["paper_width_mm"] = st.sidebar.number_input("纸面宽度 mm", 50.0, 500.0, float(st.session_state.get("paper_width_mm", DEFAULTS["paper_width_mm"])), 1.0)
    config["paper_height_mm"] = st.sidebar.number_input("纸面高度 mm", 50.0, 600.0, float(st.session_state.get("paper_height_mm", DEFAULTS["paper_height_mm"])), 1.0)
    config["cylinder_radius_mm"] = st.sidebar.slider("圆柱半径 R", 5.0, 60.0, float(st.session_state.get("cylinder_radius_mm", DEFAULTS["cylinder_radius_mm"])), 1.0)
    config["cylinder_height_mm"] = st.sidebar.slider("圆柱高度 H", 20.0, 250.0, float(st.session_state.get("cylinder_height_mm", DEFAULTS["cylinder_height_mm"])), 1.0)
    config["cylinder_center_x_mm"] = st.sidebar.slider("圆柱中心 xc", 0.0, float(config["paper_width_mm"]), float(st.session_state.get("cylinder_center_x_mm", DEFAULTS["cylinder_center_x_mm"])), 1.0)
    config["cylinder_center_y_mm"] = st.sidebar.slider("圆柱中心 yc", 0.0, float(config["paper_height_mm"]), float(st.session_state.get("cylinder_center_y_mm", DEFAULTS["cylinder_center_y_mm"])), 1.0)
    config["eye_x_mm"] = st.sidebar.slider("观察点 xe", -250.0, 450.0, float(st.session_state.get("eye_x_mm", DEFAULTS["eye_x_mm"])), 1.0)
    config["eye_y_mm"] = st.sidebar.slider("观察点 ye", -500.0, 500.0, float(st.session_state.get("eye_y_mm", DEFAULTS["eye_y_mm"])), 1.0)
    config["eye_z_mm"] = st.sidebar.slider("观察点 ze", 20.0, 800.0, float(st.session_state.get("eye_z_mm", DEFAULTS["eye_z_mm"])), 1.0)
    config["theta_span_deg"] = st.sidebar.slider("可见角范围 deg", 30.0, 180.0, float(st.session_state.get("theta_span_deg", DEFAULTS["theta_span_deg"])), 1.0)

    st.sidebar.header("离散与画布")
    config["theta_samples"] = st.sidebar.slider("theta 采样数", 60, 600, int(st.session_state.get("theta_samples", DEFAULTS["theta_samples"])), 10)
    config["height_samples"] = st.sidebar.slider("高度采样数", 60, 600, int(st.session_state.get("height_samples", DEFAULTS["height_samples"])), 10)
    config["paper_resolution_px"] = st.sidebar.slider("纸面长边分辨率 px", 300, 1400, int(st.session_state.get("paper_resolution_px", DEFAULTS["paper_resolution_px"])), 50)

    st.sidebar.header("图像适配")
    fit_modes = ["contain", "cover", "stretch"]
    config["paper_fit_mode"] = st.sidebar.selectbox("纸面图适配方式", fit_modes, index=fit_modes.index(st.session_state.get("paper_fit_mode", DEFAULTS["paper_fit_mode"])))
    config["mirror_fit_mode"] = st.sidebar.selectbox("镜面图适配方式", fit_modes, index=fit_modes.index(st.session_state.get("mirror_fit_mode", DEFAULTS["mirror_fit_mode"])))
    config["fit_paper_to_canvas"] = st.sidebar.checkbox("纸面图保持比例铺放", value=bool(st.session_state.get("fit_paper_to_canvas", DEFAULTS["fit_paper_to_canvas"])))
    config["fit_target_to_valid_region"] = st.sidebar.checkbox("目标镜面图自动铺满有效区域", value=bool(st.session_state.get("fit_target_to_valid_region", DEFAULTS["fit_target_to_valid_region"])))

    st.sidebar.header("可视化")
    config["limit_to_a4"] = st.sidebar.checkbox("统计时仅使用 A4 内映射", value=bool(st.session_state.get("limit_to_a4", DEFAULTS["limit_to_a4"])))
    config["show_out_of_bounds"] = st.sidebar.checkbox("高亮越界/无效区域", value=bool(st.session_state.get("show_out_of_bounds", DEFAULTS["show_out_of_bounds"])))
    config["show_grid"] = st.sidebar.checkbox("显示纸面坐标网格", value=bool(st.session_state.get("show_grid", DEFAULTS["show_grid"])))
    config["show_heatmap"] = st.sidebar.checkbox("显示畸变热力图", value=bool(st.session_state.get("show_heatmap", DEFAULTS["show_heatmap"])))

    for key, value in config.items():
        st.session_state[key] = value
    return mode, config


mode, config = sidebar_controls()
params = params_from_dict(config)
mapping = compute_mapping(params)
distortion = compute_distortion(mapping)
mapping_stats = summarize_mapping(mapping, params)
paper_height_px = int(config["paper_resolution_px"])
paper_width_px = int(round(config["paper_resolution_px"] * params.paper_width_mm / params.paper_height_mm))
mirror_valid_mask = mapping["inside_a4"] if config["limit_to_a4"] else mapping["valid_down"]

st.sidebar.header("图像上传")
paper_upload = st.sidebar.file_uploader("上传纸面图案", type=["png", "jpg", "jpeg"], key="paper_upload")
target_upload = st.sidebar.file_uploader("上传目标镜面图案", type=["png", "jpg", "jpeg"], key="target_upload")
default_kind = st.sidebar.selectbox("默认示例图", ["paper", "mirror", "portrait"], index=0)
paper_source_options = ["默认示例", "上传图片"]
mirror_source_options = ["默认示例", "上传图片"]

paper_default_index = 1 if (paper_upload is not None and st.session_state.get("paper_source", "默认示例") == "上传图片") else 0
mirror_default_index = 1 if (target_upload is not None and st.session_state.get("mirror_source", "默认示例") == "上传图片") else 0

paper_source = st.sidebar.radio(
    "纸面图来源",
    paper_source_options,
    index=paper_default_index,
    key="paper_source",
)
mirror_source = st.sidebar.radio(
    "镜面图来源",
    mirror_source_options,
    index=mirror_default_index,
    key="mirror_source",
)

uploaded_paper = open_uploaded_image(paper_upload)
uploaded_mirror = open_uploaded_image(target_upload)

paper_fallback_kind = "portrait" if default_kind == "portrait" else "paper"
mirror_fallback_kind = "portrait" if default_kind == "portrait" else "mirror"

if paper_source == "上传图片" and uploaded_paper is not None:
    paper_input = uploaded_paper
    paper_source_label = "上传图片"
elif paper_source == "上传图片" and uploaded_paper is None:
    st.sidebar.warning("纸面图来源已选“上传图片”，但当前没有上传文件，已自动回退到默认示例。")
    paper_input = build_default_input(paper_fallback_kind)
    paper_source_label = f"默认示例（{paper_fallback_kind}）"
else:
    paper_input = build_default_input(paper_fallback_kind)
    paper_source_label = f"默认示例（{paper_fallback_kind}）"

if mirror_source == "上传图片" and uploaded_mirror is not None:
    mirror_input = uploaded_mirror
    mirror_source_label = "上传图片"
elif mirror_source == "上传图片" and uploaded_mirror is None:
    st.sidebar.warning("镜面图来源已选“上传图片”，但当前没有上传文件，已自动回退到默认示例。")
    mirror_input = build_default_input(mirror_fallback_kind)
    mirror_source_label = f"默认示例（{mirror_fallback_kind}）"
else:
    mirror_input = build_default_input(mirror_fallback_kind)
    mirror_source_label = f"默认示例（{mirror_fallback_kind}）"

if config["fit_paper_to_canvas"]:
    paper_input = compose_image(paper_input, (paper_width_px, paper_height_px), mode=config["paper_fit_mode"], bg_color=(255, 255, 255))
else:
    paper_input = compose_image(paper_input, (paper_width_px, paper_height_px), mode="stretch", bg_color=(255, 255, 255))

if config["fit_target_to_valid_region"]:
    mirror_input = fit_image_to_mask(
        mirror_input,
        (params.theta_samples, params.height_samples),
        mask=mirror_valid_mask,
        mode=config["mirror_fit_mode"],
        bg_color=(255, 255, 255),
    )
else:
    mirror_input = compose_image(mirror_input, (params.theta_samples, params.height_samples), mode=config["mirror_fit_mode"], bg_color=(255, 255, 255))

forward_out = None
inverse_out = None
verify_forward = None
if mode in ("正向模拟", "双向验证"):
    forward_out = forward_simulation(paper_input, mapping, params, paper_width_px, paper_height_px)
if mode in ("逆向设计", "双向验证"):
    inverse_out = inverse_design(mirror_input, mapping, params, paper_width_px, paper_height_px)
    verify_forward = forward_simulation(array_to_pil(inverse_out["paper_array"]), mapping, params, paper_width_px, paper_height_px)

mirror_target_show = mirror_input
if config["show_out_of_bounds"]:
    mirror_target_show = array_to_pil(tint_invalid_regions(np.asarray(mirror_input, dtype=np.float32), mirror_valid_mask))

st.subheader("当前参数提示")
tip_cols = st.columns(4)
tip_cols[0].metric("有效反射比例", format_ratio(mapping_stats["valid_reflection_ratio"]))
tip_cols[1].metric("落入 A4 比例", format_ratio(mapping_stats["inside_a4_ratio"]))
tip_cols[2].metric("有效光线中的 A4 比例", format_ratio(mapping_stats["inside_given_valid_ratio"]))
tip_cols[3].metric("视点高于圆柱", "是" if mapping_stats["eye_above_cylinder"] > 0.5 else "否")

if mapping_stats["inside_given_valid_ratio"] < 0.6:
    st.warning("当前参数下越界较严重。建议优先提高 ze、减小 theta 角域，或将圆柱中心向纸张中部移动。")
elif mapping_stats["inside_given_valid_ratio"] < 0.85:
    st.info("当前参数可用，但仍有部分区域未落入 A4。可开启“自动铺满有效区域”提升图像利用率。")
else:
    st.success("当前参数下大部分映射落在 A4 内，适合做展示和参数实验。")

tabs = st.tabs(["主结果", "输入与适配", "诊断与热力图"])

with tabs[0]:
    if mode == "正向模拟":
        cols = st.columns(2)
        paper_show = overlay_grid(paper_input, step_px=max(25, paper_width_px // 10)) if config["show_grid"] else paper_input
        mirror_show = array_to_pil(forward_out["mirror_array"])
        if config["show_out_of_bounds"]:
            mirror_show = array_to_pil(tint_invalid_regions(forward_out["mirror_array"], mapping["inside_a4"]))
        cols[0].image(paper_show, caption="输入纸面图案", use_container_width=True)
        cols[1].image(mirror_show, caption="正向模拟镜面图案", use_container_width=True)
    elif mode == "逆向设计":
        cols = st.columns(3)
        cols[0].image(mirror_target_show, caption="目标镜面图案", use_container_width=True)
        designed_paper = array_to_pil(inverse_out["paper_array"])
        designed_show = overlay_grid(designed_paper, step_px=max(25, paper_width_px // 10)) if config["show_grid"] else designed_paper
        designed_show = draw_bbox_overlay(designed_show, paper_usage_bbox(inverse_out["paper_weight"]))
        verify_show = array_to_pil(verify_forward["mirror_array"])
        if config["show_out_of_bounds"]:
            verify_show = array_to_pil(tint_invalid_regions(verify_forward["mirror_array"], mapping["inside_a4"]))
        cols[1].image(designed_show, caption="反求纸面图案", use_container_width=True)
        cols[2].image(verify_show, caption="正向回代验证", use_container_width=True)
    else:
        top = st.columns(3)
        top[0].image(paper_input, caption="原始纸面图案", use_container_width=True)
        top[1].image(mirror_target_show, caption="目标镜面图案", use_container_width=True)
        forward_show = array_to_pil(forward_out["mirror_array"])
        if config["show_out_of_bounds"]:
            forward_show = array_to_pil(tint_invalid_regions(forward_out["mirror_array"], mapping["inside_a4"]))
        top[2].image(forward_show, caption="原纸面正向镜面图", use_container_width=True)
        bottom = st.columns(2)
        designed_paper = array_to_pil(inverse_out["paper_array"])
        designed_show = draw_bbox_overlay(designed_paper, paper_usage_bbox(inverse_out["paper_weight"]))
        verify_show = array_to_pil(verify_forward["mirror_array"])
        if config["show_out_of_bounds"]:
            verify_show = array_to_pil(tint_invalid_regions(verify_forward["mirror_array"], mapping["inside_a4"]))
        bottom[0].image(designed_show, caption="反求得到的纸面图案", use_container_width=True)
        bottom[1].image(verify_show, caption="双向验证镜面图", use_container_width=True)

with tabs[1]:
    info_cols = st.columns(2)
    info_cols[0].info(f"当前纸面图来源：{paper_source_label}")
    info_cols[1].info(f"当前镜面图来源：{mirror_source_label}")
    cols = st.columns(3)
    cols[0].image(paper_input, caption="纸面输入画布", use_container_width=True)
    cols[1].image(mirror_target_show, caption="镜面参数域目标图", use_container_width=True)
    cols[2].image(make_heatmap(mirror_valid_mask.astype(float)), caption="有效镜面参数域", use_container_width=True)
    st.markdown(
        "<div class='small-note'>已增加“保持比例铺放”和“自动铺满有效区域”两类适配模式。前者避免上传图被强制拉伸，后者让镜面目标优先利用真正可映射的有效区域，修复了之前图片利用不充分的主要问题。</div>",
        unsafe_allow_html=True,
    )

with tabs[2]:
    diag = st.columns(4)
    diag[0].metric("纸面映射 x 范围", f"{format_float(mapping_stats['paper_bbox_xmin_mm'],1)} ~ {format_float(mapping_stats['paper_bbox_xmax_mm'],1)} mm")
    diag[1].metric("纸面映射 y 范围", f"{format_float(mapping_stats['paper_bbox_ymin_mm'],1)} ~ {format_float(mapping_stats['paper_bbox_ymax_mm'],1)} mm")
    diag[2].metric("镜面分辨率", f"{params.theta_samples} × {params.height_samples}")
    if inverse_out is not None:
        usage_ratio = float((inverse_out["paper_weight"] > 1e-6).mean())
        diag[3].metric("纸面利用率", format_ratio(usage_ratio))
    else:
        diag[3].metric("纸面利用率", "N/A")

    if config["show_heatmap"]:
        heat_cols = st.columns(2)
        heat_cols[0].image(make_heatmap(distortion, invalid_mask=~mapping["inside_a4"]), caption="局部畸变热力图", use_container_width=True)
        density_img = make_heatmap(inverse_out["paper_weight"]) if inverse_out is not None else make_heatmap(mapping["inside_a4"].astype(float))
        heat_cols[1].image(density_img, caption="覆盖密度 / 映射权重图", use_container_width=True)

st.subheader("数值指标")
dist_stats = distortion_stats(distortion)
metric_cols = st.columns(6)
metric_cols[0].metric("映射有效比例", format_ratio(mapping_stats["valid_reflection_ratio"]))
metric_cols[1].metric("A4 内比例", format_ratio(mapping_stats["inside_a4_ratio"]))
metric_cols[2].metric("畸变均值", format_float(dist_stats["distortion_mean"], 2))
metric_cols[3].metric("畸变 P95", format_float(dist_stats["distortion_p95"], 2))
metric_cols[4].metric("畸变最大值", format_float(dist_stats["distortion_max"], 2))
metric_cols[5].metric("角域跨度", f"{format_float(params.theta_span_deg,1)} deg")

if verify_forward is not None:
    err_cols = st.columns(3)
    mask = mapping["inside_a4"]
    err_cols[0].metric("镜面重建 MSE", format_float(mse(verify_forward["mirror_array"], inverse_out["target_array"], mask=mask), 3))
    err_cols[1].metric("PSNR", format_float(psnr(verify_forward["mirror_array"], inverse_out["target_array"], mask=mask), 3))
    err_cols[2].metric("SSIM 近似", format_float(ssim_approx(verify_forward["mirror_array"], inverse_out["target_array"], mask=mask), 4))

st.subheader("导出")
export_cols = st.columns(3)
if inverse_out is not None:
    export_cols[0].download_button("导出纸面图案 PNG", data=image_to_bytes(array_to_pil(inverse_out["paper_array"])), file_name="designed_paper.png", mime="image/png", use_container_width=True)
if forward_out is not None:
    export_cols[1].download_button("导出正向镜面图 PNG", data=image_to_bytes(array_to_pil(forward_out["mirror_array"])), file_name="forward_mirror.png", mime="image/png", use_container_width=True)
if verify_forward is not None:
    export_cols[2].download_button("导出验证镜面图 PNG", data=image_to_bytes(array_to_pil(verify_forward["mirror_array"])), file_name="validated_mirror.png", mime="image/png", use_container_width=True)

with st.expander("模型说明"):
    st.markdown(
        """
        - 严格几何部分：圆柱面参数化、法向量、镜面反射定律，以及反射光线与 `z=0` 的求交。
        - 数值离散部分：将镜面离散为 `theta × h` 网格，通过双线性采样与散点回填完成正向/逆向映射。
        - 本次优化新增：上传图保持比例铺放、目标图自动对齐有效参数域、越界高亮、纸面利用率边框提示，以及更清晰的多标签界面。
        """
    )
