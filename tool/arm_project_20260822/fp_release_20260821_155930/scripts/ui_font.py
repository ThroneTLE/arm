#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中文字体工具：给 Tk / ttk 界面统一设置 CJK 字体。

用法:
    import ui_font
    fam = ui_font.setup_cn_font(root, size=11)

说明:
    - ttk 控件（Button/Combobox/Entry/Label 等）不吃 option_add，
      必须同时配置 ttk.Style，这里一并处理。
    - 如果系统里没有任何中文字体，会打印安装提示：
      sudo apt update && sudo apt install -y fonts-noto-cjk
"""

import os
import subprocess
import tkinter.font as tkfont

# conda 里的 Tk 自带 fontconfig，默认看不到系统 /usr/share/fonts 的字体。
# 在 Tk 初始化（tk.Tk()）之前把它指到系统配置，否则中文字体列表是空的。
if os.path.exists("/etc/fonts/fonts.conf"):
    os.environ.setdefault("FONTCONFIG_FILE", "/etc/fonts/fonts.conf")
    os.environ.setdefault("FONTCONFIG_PATH", "/etc/fonts")

# ============ 可调配置（想换字体改这里） ============
# 指定字体族名；留空=自动挑选。
# 常见可选:
#   Noto Sans CJK SC      （已装 fonts-noto-cjk）
#   WenQuanYi Micro Hei   （安装: sudo apt install -y fonts-wqy-microhei）
#   WenQuanYi Zen Hei     （安装: sudo apt install -y fonts-wqy-zenhei）
UI_FONT_FAMILY = "WenQuanYi Micro Hei"
UI_FONT_SIZE = 12

# 优先按顺序找这些字体族
PREFERRED = [
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Noto Sans SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "PingFang SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "AR PL UMing CN",
    "Droid Sans Fallback",
]

# 兜底：按关键字扫描所有可用字体
KEYWORDS = ("cjk", "yahei", "wenquanyi", "wqy", "simhei", "simsun",
            "pingfang", "source han", "noto sans sc", "micro hei",
            "zen hei", "droid sans fallback", "uming", "ukai")


def find_cn_font(root=None):
    """返回一个可用的中文字体族名；找不到返回 None。"""
    try:
        fams = set(tkfont.families(root))
    except Exception:
        fams = set()
    if UI_FONT_FAMILY:
        if UI_FONT_FAMILY in fams:
            return UI_FONT_FAMILY
        print(f"[字体] 指定的 {UI_FONT_FAMILY} 未安装，自动改用其他中文字体")
    for name in PREFERRED:
        if name in fams:
            return name
    lower = {f.lower(): f for f in fams}
    for kw in KEYWORDS:
        for fl, real in lower.items():
            if kw in fl:
                return real
    # Tk 列表不可靠时，直接问 fontconfig（Linux 下 Tk 走 Xft，认 fontconfig 字体）
    fam = _fc_match_zh()
    if fam:
        return fam
    return None


def _fc_match_zh():
    """用 fontconfig 查一个中文字体族名，例如 'Noto Sans CJK SC'。"""
    for args in (["fc-match", "-f", "%{family}", ":lang=zh"],
                 ["fc-match", "-f", "%{family}", "sans-serif"]):
        try:
            out = subprocess.check_output(args, text=True, timeout=5).strip()
        except Exception:
            continue
        if out:
            return out.split(",")[0].strip()
    return None


def setup_cn_font(root, size=None):
    """设置全局中文字体；返回选中的字体族（没有中文字体则返回 None）。"""
    if size is None:
        size = UI_FONT_SIZE
    fam = find_cn_font(root)
    if fam is None:
        try:
            fams_now = sorted(tkfont.families(root))
        except Exception:
            fams_now = []
        print("[字体] Tk 当前共读到 %d 个字体族，示例: %s"
              % (len(fams_now), fams_now[:10]))
        print("[字体] 未找到中文字体，界面中文可能显示为方块。")
        print("[字体] 请安装：sudo apt update && sudo apt install -y fonts-noto-cjk")
        print("[字体] 检查是否已装：fc-list :lang=zh family | head")
        return None

    # 1) Tk 控件（tk.Label / tk.Button / tk.Entry 等）走 option database
    root.option_add("*Font", "{%s} %d" % (fam, size))

    # 2) ttk 控件（ttk.Button / ttk.Combobox / ttk.Entry 等）走 ttk.Style
    try:
        from tkinter import ttk
        style = ttk.Style(root)
        style.configure(".", font=(fam, size))
        for cls in ("TButton", "TLabel", "TEntry", "TCombobox",
                    "TFrame", "TLabelframe", "TLabelframe.Label",
                    "TNotebook", "TCheckbutton", "TRadiobutton",
                    "TMenubutton", "TSpinbox", "Treeview", "Treeview.Heading"):
            try:
                style.configure(cls, font=(fam, size))
            except Exception:
                pass
    except Exception:
        pass

    print("[字体] 使用中文字体:", fam, "| 字号", size)
    return fam
