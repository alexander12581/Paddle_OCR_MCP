# PaddleOCR `predict()` 全部返回字段详解

**模型**: PP-OCRv6 medium  |  **设备**: RTX 4070 Ti SUPER (GPU)  |  **时间**: 2026-06-28 19:07:12

---

## 字段总览

`predict()` 返回 `list[dict]`，每页一个 dict，含以下 15 个字段：

### 核心识别
| 字段 | 类型 | 说明 |
|------|------|------|
| `rec_texts` | `list[str]` | 识别出的文字内容 |
| `rec_scores` | `list[float]` | 置信度 (0~1)，越高越可信 |
| `dt_polys` | `list[ndarray(4,2)]` | 文本检测框，4 个角点坐标 `[[x0,y0],[x1,y1],[x2,y2],[x3,y3]]` |
| `rec_polys` | `list[ndarray(4,2)]` | 文本识别框（矫正后），比检测框更精确 |
| `rec_boxes` | `ndarray(N,4)` | 紧凑包围盒 `(x, y, w, h)` — 左上角+宽高 |

### 文档预处理
| 字段 | 类型 | 说明 |
|------|------|------|
| `doc_preprocessor_res` | `dict` | 文档预处理结果：含原始图像、矫正图像、旋转角度、是否触发矫正等 |
| `textline_orientation_angles` | `list[int]` | 每行文字的方向角：`0`=正向 `90`=右旋 `180`=倒置 `270`=左旋 |
| `model_settings` | `dict` | 记录预处理开关状态 |

### 检测参数
| 字段 | 类型 | 说明 |
|------|------|------|
| `text_det_params` | `dict` | 文本检测参数：边长限制、阈值、框阈值等 |
| `text_type` | `str` | 文本类型，固定为 `"general"` |
| `text_rec_score_thresh` | `float` | 识别分数阈值，低于此值的文字会被丢弃 |
| `return_word_box` | `bool` | 是否返回单词级别包围框（默认 False） |

### 元信息
| 字段 | 类型 | 说明 |
|------|------|------|
| `input_path` | `str` | 输入图片的完整路径 |
| `page_index` | `None\|int` | PDF 的页码，图片为 None |
| `vis_fonts` | `list[Font]` | 可视化用的字体对象列表（用于渲染识别结果） |

---

## 图1. Snipaste_2026-06-27_09-58-43.png

![Snipaste_2026-06-27_09-58-43.png](Snipaste_2026-06-27_09-58-43.png)
*耗时 0.9s，识别 39 条文字*

### `input_path` — 输入路径

```
E:\soft\OCR\Paddle_ocr\Snipaste_2026-06-27_09-58-43.png
```

### `page_index` — 页码

```
None
```
> 图片为 `None`，PDF 多页时为页码序号

### `doc_preprocessor_res` — 文档预处理

- **_save_funcs**: [<weakref at 0x00000212E1625D50; to 'DocPreprocessorResult' at 0x00000212E1604870>, <weakref at 0x00000212E16260D0; to 'DocPreprocessorResult' at 0x00000212E1604870>]
- **_json_writer**: <paddlex.inference.utils.io.writers.JsonWriter object at 0x00000212E15EB020>
- **_rand_fn**: None
- **_img_writer**: <paddlex.inference.utils.io.writers.ImageWriter object at 0x00000212E15EA5D0>

### `dt_polys` — 检测框坐标

共 `39` 个框，每个是 `(4,2)` 的 numpy 数组（4个角点 x,y）：

- 框[0]: `[[0, 0], [380, 3], [379, 44], [0, 31]]` → 文字: `arrange(start,end,step)`
- 框[1]: `[[597, 28], [732, 28], [732, 70], [597, 70]]` → 文字: `搭建网络`
- 框[2]: `[[1328, 50], [1669, 108], [1646, 241], [1305, 183]]` → 文字: `w-naw`
- ... 共 39 个

### `model_settings` — 模型设置

```json
{
  "use_doc_preprocessor": true,
  "use_textline_orientation": true
}
```

### `text_det_params` — 文本检测参数

```json
{
  "limit_side_len": 64,
  "limit_type": "min",
  "thresh": 0.3,
  "max_side_limit": 4000,
  "box_thresh": 0.6,
  "unclip_ratio": 1.5
}
```

### `text_type` — 文本类型

```
general
```

### `text_rec_score_thresh` — 识别分数阈值

```
0.0
```
> 低于此值的识别结果被丢弃。`0.0` 表示不过滤

### `return_word_box` — 是否返回词级包围框

```
False
```

### `rec_texts` / `rec_scores` / `rec_polys` — 识别三件套

共 `39` 条。前 10 条：

| # | rec_texts | rec_scores | rec_polys (左上角) |
|---|-----------|------------|---------------------|
| 0 | arrange(start,end,step) | 100.0% | (0,0) |
| 1 | 搭建网络 | 100.0% | (597,28) |
| 2 | w-naw | 74.4% | (1328,50) |
| 3 | aL | 89.4% | (1516,42) |
| 4 | 12原则 | 100.0% | (871,53) |
| 5 | [ | 90.9% | (78,81) |
| 6 | )左闭右开 | 99.9% | (151,79) |
| 7 | 1.1个继承 nn.Module | 98.9% | (592,123) |
| 8 | 2.2个方法 init | 98.6% | (589,209) |
| 9 | forward | 100.0% | (892,212) |
| ... | ... | ... | ... |

### `vis_fonts` — 可视化字体

共 `39` 个 Font 对象，用于绘制识别结果

### `textline_orientation_angles` — 文本行方向角

全 `39` 行，取值分布: {0: 39}

> `0`=正向 `90`=右旋 `180`=倒置 `270`=左旋

### `rec_boxes` — 包围盒 (x,y,w,h)

ndarray shape=`(39, 4)` dtype=`int16`

前 5 行 `[x, y, w, h]`：

| # | x | y | w | h | 对应文字 |
|---|---|---|---|---|----------|
| 0 | 0 | 0 | 380 | 44 | arrange(start,end,step) |
| 1 | 597 | 28 | 732 | 70 | 搭建网络 |
| 2 | 1305 | 50 | 1669 | 241 | w-naw |
| 3 | 1516 | 42 | 1629 | 123 | aL |
| 4 | 870 | 53 | 982 | 103 | 12原则 |

---

## 图2. Snipaste_2026-06-27_09-59-11.png

![Snipaste_2026-06-27_09-59-11.png](Snipaste_2026-06-27_09-59-11.png)
*耗时 1.1s，识别 86 条文字*

### `input_path` — 输入路径

```
E:\soft\OCR\Paddle_ocr\Snipaste_2026-06-27_09-59-11.png
```

### `page_index` — 页码

```
None
```
> 图片为 `None`，PDF 多页时为页码序号

### `doc_preprocessor_res` — 文档预处理

- **_save_funcs**: [<weakref at 0x00000212E1633AD0; to 'DocPreprocessorResult' at 0x00000212E1605EF0>, <weakref at 0x00000212E1633550; to 'DocPreprocessorResult' at 0x00000212E1605EF0>]
- **_json_writer**: <paddlex.inference.utils.io.writers.JsonWriter object at 0x00000213277A93D0>
- **_rand_fn**: None
- **_img_writer**: <paddlex.inference.utils.io.writers.ImageWriter object at 0x00000213277A9430>

### `dt_polys` — 检测框坐标

共 `86` 个框，每个是 `(4,2)` 的 numpy 数组（4个角点 x,y）：

- 框[0]: `[[0, 1], [24, 1], [24, 25], [0, 25]]` → 文字: `和`
- 框[1]: `[[519, 2], [541, 2], [541, 14], [519, 14]]` → 文字: `0`
- 框[2]: `[[560, 1], [586, 1], [586, 12], [560, 12]]` → 文字: `°□`
- ... 共 86 个

### `model_settings` — 模型设置

```json
{
  "use_doc_preprocessor": true,
  "use_textline_orientation": true
}
```

### `text_det_params` — 文本检测参数

```json
{
  "limit_side_len": 64,
  "limit_type": "min",
  "thresh": 0.3,
  "max_side_limit": 4000,
  "box_thresh": 0.6,
  "unclip_ratio": 1.5
}
```

### `text_type` — 文本类型

```
general
```

### `text_rec_score_thresh` — 识别分数阈值

```
0.0
```
> 低于此值的识别结果被丢弃。`0.0` 表示不过滤

### `return_word_box` — 是否返回词级包围框

```
False
```

### `rec_texts` / `rec_scores` / `rec_polys` — 识别三件套

共 `86` 条。前 10 条：

| # | rec_texts | rec_scores | rec_polys (左上角) |
|---|-----------|------------|---------------------|
| 0 | 和 | 15.8% | (0,1) |
| 1 | 0 | 46.5% | (519,2) |
| 2 | °□ | 41.9% | (560,1) |
| 3 | 十  | 55.0% | (762,1) |
| 4 | Momentum | 100.0% | (250,27) |
| 5 | KW | 64.1% | (441,28) |
| 6 | Adam | 100.0% | (1356,42) |
| 7 | BP算法 | 100.0% | (1042,75) |
| 8 | optimizer = torch.optim.SCD([w], Ir=0.01, momentum | 99.1% | (413,126) |
| 9 | 反向传播算法 | 100.0% | (1039,129) |
| ... | ... | ... | ... |

### `vis_fonts` — 可视化字体

共 `86` 个 Font 对象，用于绘制识别结果

### `textline_orientation_angles` — 文本行方向角

全 `86` 行，取值分布: {0: 86}

> `0`=正向 `90`=右旋 `180`=倒置 `270`=左旋

### `rec_boxes` — 包围盒 (x,y,w,h)

ndarray shape=`(86, 4)` dtype=`int16`

前 5 行 `[x, y, w, h]`：

| # | x | y | w | h | 对应文字 |
|---|---|---|---|---|----------|
| 0 | 0 | 1 | 24 | 25 | 和 |
| 1 | 519 | 2 | 541 | 14 | 0 |
| 2 | 560 | 1 | 586 | 12 | °□ |
| 3 | 762 | 1 | 813 | 19 | 十  |
| 4 | 249 | 27 | 350 | 53 | Momentum |

---

## 图3. Snipaste_2026-06-27_09-59-21.png

![Snipaste_2026-06-27_09-59-21.png](Snipaste_2026-06-27_09-59-21.png)
*耗时 0.6s，识别 53 条文字*

### `input_path` — 输入路径

```
E:\soft\OCR\Paddle_ocr\Snipaste_2026-06-27_09-59-21.png
```

### `page_index` — 页码

```
None
```
> 图片为 `None`，PDF 多页时为页码序号

### `doc_preprocessor_res` — 文档预处理

- **_save_funcs**: [<weakref at 0x00000212F4319AD0; to 'DocPreprocessorResult' at 0x00000212F4328190>, <weakref at 0x00000212F431AF50; to 'DocPreprocessorResult' at 0x00000212F4328190>]
- **_json_writer**: <paddlex.inference.utils.io.writers.JsonWriter object at 0x00000212E162CB60>
- **_rand_fn**: None
- **_img_writer**: <paddlex.inference.utils.io.writers.ImageWriter object at 0x00000212E162C230>

### `dt_polys` — 检测框坐标

共 `53` 个框，每个是 `(4,2)` 的 numpy 数组（4个角点 x,y）：

- 框[0]: `[[31, 0], [93, 2], [92, 31], [30, 29]]` → 文字: `已编辑`
- 框[1]: `[[573, 1], [657, 1], [657, 37], [573, 37]]` → 文字: `6`
- 框[2]: `[[678, 1], [941, 0], [942, 34], [679, 38]]` → 文字: ` } □ +  step()`
- ... 共 53 个

### `model_settings` — 模型设置

```json
{
  "use_doc_preprocessor": true,
  "use_textline_orientation": true
}
```

### `text_det_params` — 文本检测参数

```json
{
  "limit_side_len": 64,
  "limit_type": "min",
  "thresh": 0.3,
  "max_side_limit": 4000,
  "box_thresh": 0.6,
  "unclip_ratio": 1.5
}
```

### `text_type` — 文本类型

```
general
```

### `text_rec_score_thresh` — 识别分数阈值

```
0.0
```
> 低于此值的识别结果被丢弃。`0.0` 表示不过滤

### `return_word_box` — 是否返回词级包围框

```
False
```

### `rec_texts` / `rec_scores` / `rec_polys` — 识别三件套

共 `53` 条。前 10 条：

| # | rec_texts | rec_scores | rec_polys (左上角) |
|---|-----------|------------|---------------------|
| 0 | 已编辑 | 99.6% | (31,0) |
| 1 | 6 | 24.5% | (573,1) |
| 2 |  } □ +  step() | 71.4% | (678,1) |
| 3 |  | 0.0% | (1211,1) |
| 4 | C  □ | 55.1% | (1278,2) |
| 5 | ① | 40.2% | (1435,3) |
| 6 | 日 | 44.2% | (1470,4) |
| 7 | 升级 | 100.0% | (1544,4) |
| 8 | □ | 61.6% | (1705,1) |
| 9 | X | 90.1% | (1774,0) |
| ... | ... | ... | ... |

### `vis_fonts` — 可视化字体

共 `53` 个 Font 对象，用于绘制识别结果

### `textline_orientation_angles` — 文本行方向角

全 `53` 行，取值分布: {0: 53}

> `0`=正向 `90`=右旋 `180`=倒置 `270`=左旋

### `rec_boxes` — 包围盒 (x,y,w,h)

ndarray shape=`(53, 4)` dtype=`int16`

前 5 行 `[x, y, w, h]`：

| # | x | y | w | h | 对应文字 |
|---|---|---|---|---|----------|
| 0 | 30 | 0 | 93 | 31 | 已编辑 |
| 1 | 573 | 1 | 657 | 37 | 6 |
| 2 | 678 | 0 | 942 | 38 |  } □ +  step() |
| 3 | 1211 | 1 | 1249 | 39 |  |
| 4 | 1278 | 2 | 1400 | 37 | C  □ |

---

## 图4. Snipaste_2026-06-27_09-59-33.png

![Snipaste_2026-06-27_09-59-33.png](Snipaste_2026-06-27_09-59-33.png)
*耗时 0.2s，识别 11 条文字*

### `input_path` — 输入路径

```
E:\soft\OCR\Paddle_ocr\Snipaste_2026-06-27_09-59-33.png
```

### `page_index` — 页码

```
None
```
> 图片为 `None`，PDF 多页时为页码序号

### `doc_preprocessor_res` — 文档预处理

- **_save_funcs**: [<weakref at 0x00000212F43197D0; to 'DocPreprocessorResult' at 0x00000212F4328230>, <weakref at 0x00000212F4319850; to 'DocPreprocessorResult' at 0x00000212F4328230>]
- **_json_writer**: <paddlex.inference.utils.io.writers.JsonWriter object at 0x00000212F433C680>
- **_rand_fn**: None
- **_img_writer**: <paddlex.inference.utils.io.writers.ImageWriter object at 0x00000212F433C050>

### `dt_polys` — 检测框坐标

共 `11` 个框，每个是 `(4,2)` 的 numpy 数组（4个角点 x,y）：

- 框[0]: `[[33, 58], [182, 58], [182, 125], [33, 125]]` → 文字: `GAN`
- 框[1]: `[[318, 218], [512, 218], [512, 275], [318, 275]]` → 文字: `以假乱真`
- 框[2]: `[[30, 315], [252, 315], [252, 394], [30, 394]]` → 文字: `生成器`
- ... 共 11 个

### `model_settings` — 模型设置

```json
{
  "use_doc_preprocessor": true,
  "use_textline_orientation": true
}
```

### `text_det_params` — 文本检测参数

```json
{
  "limit_side_len": 64,
  "limit_type": "min",
  "thresh": 0.3,
  "max_side_limit": 4000,
  "box_thresh": 0.6,
  "unclip_ratio": 1.5
}
```

### `text_type` — 文本类型

```
general
```

### `text_rec_score_thresh` — 识别分数阈值

```
0.0
```
> 低于此值的识别结果被丢弃。`0.0` 表示不过滤

### `return_word_box` — 是否返回词级包围框

```
False
```

### `rec_texts` / `rec_scores` / `rec_polys` — 识别三件套

共 `11` 条。前 10 条：

| # | rec_texts | rec_scores | rec_polys (左上角) |
|---|-----------|------------|---------------------|
| 0 | GAN | 100.0% | (33,58) |
| 1 | 以假乱真 | 100.0% | (318,218) |
| 2 | 生成器 | 100.0% | (30,315) |
| 3 | 判别器 | 97.8% | (571,323) |
| 4 | M-F+2P | 98.9% | (1259,438) |
| 5 | 生成图片 | 100.0% | (35,457) |
| 6 | 真实 | 100.0% | (569,453) |
| 7 | N= | 92.9% | (1000,496) |
| 8 | 生成-假的 | 100.0% | (570,510) |
| 9 | S | 100.0% | (1365,566) |
| ... | ... | ... | ... |

### `vis_fonts` — 可视化字体

共 `11` 个 Font 对象，用于绘制识别结果

### `textline_orientation_angles` — 文本行方向角

全 `11` 行，取值分布: {0: 11}

> `0`=正向 `90`=右旋 `180`=倒置 `270`=左旋

### `rec_boxes` — 包围盒 (x,y,w,h)

ndarray shape=`(11, 4)` dtype=`int16`

前 5 行 `[x, y, w, h]`：

| # | x | y | w | h | 对应文字 |
|---|---|---|---|---|----------|
| 0 | 33 | 58 | 182 | 125 | GAN |
| 1 | 318 | 218 | 512 | 275 | 以假乱真 |
| 2 | 30 | 315 | 252 | 394 | 生成器 |
| 3 | 570 | 323 | 774 | 405 | 判别器 |
| 4 | 1259 | 436 | 1692 | 567 | M-F+2P |

---

## 图5. Snipaste_2026-06-27_09-59-39.png

![Snipaste_2026-06-27_09-59-39.png](Snipaste_2026-06-27_09-59-39.png)
*耗时 0.3s，识别 29 条文字*

### `input_path` — 输入路径

```
E:\soft\OCR\Paddle_ocr\Snipaste_2026-06-27_09-59-39.png
```

### `page_index` — 页码

```
None
```
> 图片为 `None`，PDF 多页时为页码序号

### `doc_preprocessor_res` — 文档预处理

- **_save_funcs**: [<weakref at 0x00000212F431BE50; to 'DocPreprocessorResult' at 0x00000212F432A620>, <weakref at 0x00000212F431A4D0; to 'DocPreprocessorResult' at 0x00000212F432A620>]
- **_json_writer**: <paddlex.inference.utils.io.writers.JsonWriter object at 0x00000212F433D670>
- **_rand_fn**: None
- **_img_writer**: <paddlex.inference.utils.io.writers.ImageWriter object at 0x00000212F433D520>

### `dt_polys` — 检测框坐标

共 `29` 个框，每个是 `(4,2)` 的 numpy 数组（4个角点 x,y）：

- 框[0]: `[[190, 0], [413, 0], [413, 41], [189, 38]]` → 文字: `Embed dim`
- 框[1]: `[[1320, 15], [1670, 13], [1671, 61], [1320, 63]]` → 文字: `深度学习模型训练`
- 框[2]: `[[903, 28], [1076, 28], [1076, 81], [903, 81]]` → 文字: `循环单元`
- ... 共 29 个

### `model_settings` — 模型设置

```json
{
  "use_doc_preprocessor": true,
  "use_textline_orientation": true
}
```

### `text_det_params` — 文本检测参数

```json
{
  "limit_side_len": 64,
  "limit_type": "min",
  "thresh": 0.3,
  "max_side_limit": 4000,
  "box_thresh": 0.6,
  "unclip_ratio": 1.5
}
```

### `text_type` — 文本类型

```
general
```

### `text_rec_score_thresh` — 识别分数阈值

```
0.0
```
> 低于此值的识别结果被丢弃。`0.0` 表示不过滤

### `return_word_box` — 是否返回词级包围框

```
False
```

### `rec_texts` / `rec_scores` / `rec_polys` — 识别三件套

共 `29` 条。前 10 条：

| # | rec_texts | rec_scores | rec_polys (左上角) |
|---|-----------|------------|---------------------|
| 0 | Embed dim | 100.0% | (190,0) |
| 1 | 深度学习模型训练 | 100.0% | (1320,15) |
| 2 | 循环单元 | 100.0% | (903,28) |
| 3 | 词向量维度 | 100.0% | (191,38) |
| 4 | RNN | 100.0% | (653,38) |
| 5 | 我 | 100.0% | (0,89) |
| 6 | 225原则 | 100.0% | (1428,95) |
| 7 | 爱 | 99.5% | (0,137) |
| 8 | 北京 | 100.0% | (0,183) |
| 9 | LSTM | 100.0% | (658,202) |
| ... | ... | ... | ... |

### `vis_fonts` — 可视化字体

共 `29` 个 Font 对象，用于绘制识别结果

### `textline_orientation_angles` — 文本行方向角

全 `29` 行，取值分布: {0: 29}

> `0`=正向 `90`=右旋 `180`=倒置 `270`=左旋

### `rec_boxes` — 包围盒 (x,y,w,h)

ndarray shape=`(29, 4)` dtype=`int16`

前 5 行 `[x, y, w, h]`：

| # | x | y | w | h | 对应文字 |
|---|---|---|---|---|----------|
| 0 | 189 | 0 | 413 | 41 | Embed dim |
| 1 | 1320 | 13 | 1671 | 63 | 深度学习模型训练 |
| 2 | 903 | 28 | 1076 | 81 | 循环单元 |
| 3 | 190 | 38 | 387 | 92 | 词向量维度 |
| 4 | 653 | 38 | 765 | 90 | RNN |

---

## 图6. Snipaste_2026-06-27_09-59-48.png

![Snipaste_2026-06-27_09-59-48.png](Snipaste_2026-06-27_09-59-48.png)
*耗时 1.0s，识别 120 条文字*

### `input_path` — 输入路径

```
E:\soft\OCR\Paddle_ocr\Snipaste_2026-06-27_09-59-48.png
```

### `page_index` — 页码

```
None
```
> 图片为 `None`，PDF 多页时为页码序号

### `doc_preprocessor_res` — 文档预处理

- **_save_funcs**: [<weakref at 0x00000212F431B350; to 'DocPreprocessorResult' at 0x00000212F432BC00>, <weakref at 0x00000212F431AB50; to 'DocPreprocessorResult' at 0x00000212F432BC00>]
- **_json_writer**: <paddlex.inference.utils.io.writers.JsonWriter object at 0x00000212F433EA50>
- **_rand_fn**: None
- **_img_writer**: <paddlex.inference.utils.io.writers.ImageWriter object at 0x00000212F433CCE0>

### `dt_polys` — 检测框坐标

共 `120` 个框，每个是 `(4,2)` 的 numpy 数组（4个角点 x,y）：

- 框[0]: `[[0, 0], [374, 8], [373, 42], [0, 32]]` → 文字: `2017谷歌颠覆性提出了基于`
- 框[1]: `[[462, 0], [541, 2], [540, 27], [461, 25]]` → 文字: `Softmax`
- 框[2]: `[[737, 0], [890, 0], [891, 55], [738, 59]]` → 文字: `BERT`
- ... 共 120 个

### `model_settings` — 模型设置

```json
{
  "use_doc_preprocessor": true,
  "use_textline_orientation": true
}
```

### `text_det_params` — 文本检测参数

```json
{
  "limit_side_len": 64,
  "limit_type": "min",
  "thresh": 0.3,
  "max_side_limit": 4000,
  "box_thresh": 0.6,
  "unclip_ratio": 1.5
}
```

### `text_type` — 文本类型

```
general
```

### `text_rec_score_thresh` — 识别分数阈值

```
0.0
```
> 低于此值的识别结果被丢弃。`0.0` 表示不过滤

### `return_word_box` — 是否返回词级包围框

```
False
```

### `rec_texts` / `rec_scores` / `rec_polys` — 识别三件套

共 `120` 条。前 10 条：

| # | rec_texts | rec_scores | rec_polys (左上角) |
|---|-----------|------------|---------------------|
| 0 | 2017谷歌颠覆性提出了基于 | 99.9% | (0,0) |
| 1 | Softmax | 100.0% | (462,0) |
| 2 | BERT | 100.0% | (737,0) |
| 3 | 12层-Encoder | 100.0% | (1044,20) |
| 4 | 自注意力机制的Transformer | 100.0% | (1,36) |
| 5 | Linear | 100.0% | (470,58) |
| 6 | 12head | 100.0% | (1124,82) |
| 7 | QKT | 95.8% | (100,106) |
| 8 | 3 + 2 + 12 + 12 +768 | 94.7% | (744,115) |
| 9 | Add & norm | 97.7% | (447,128) |
| ... | ... | ... | ... |

### `vis_fonts` — 可视化字体

共 `120` 个 Font 对象，用于绘制识别结果

### `textline_orientation_angles` — 文本行方向角

全 `120` 行，取值分布: {0: 120}

> `0`=正向 `90`=右旋 `180`=倒置 `270`=左旋

### `rec_boxes` — 包围盒 (x,y,w,h)

ndarray shape=`(120, 4)` dtype=`int16`

前 5 行 `[x, y, w, h]`：

| # | x | y | w | h | 对应文字 |
|---|---|---|---|---|----------|
| 0 | 0 | 0 | 374 | 42 | 2017谷歌颠覆性提出了基于 |
| 1 | 461 | 0 | 541 | 27 | Softmax |
| 2 | 737 | 0 | 891 | 59 | BERT |
| 3 | 1044 | 16 | 1287 | 62 | 12层-Encoder |
| 4 | 0 | 36 | 385 | 77 | 自注意力机制的Transformer |

---

## 图7. Snipaste_2026-06-27_10-00-02.png

![Snipaste_2026-06-27_10-00-02.png](Snipaste_2026-06-27_10-00-02.png)
*耗时 0.7s，识别 72 条文字*

### `input_path` — 输入路径

```
E:\soft\OCR\Paddle_ocr\Snipaste_2026-06-27_10-00-02.png
```

### `page_index` — 页码

```
None
```
> 图片为 `None`，PDF 多页时为页码序号

### `doc_preprocessor_res` — 文档预处理

- **_save_funcs**: [<weakref at 0x00000212E1626E50; to 'DocPreprocessorResult' at 0x00000212E16072A0>, <weakref at 0x00000212E1625F50; to 'DocPreprocessorResult' at 0x00000212E16072A0>]
- **_json_writer**: <paddlex.inference.utils.io.writers.JsonWriter object at 0x00000212E162CE60>
- **_rand_fn**: None
- **_img_writer**: <paddlex.inference.utils.io.writers.ImageWriter object at 0x00000212E162D310>

### `dt_polys` — 检测框坐标

共 `72` 个框，每个是 `(4,2)` 的 numpy 数组（4个角点 x,y）：

- 框[0]: `[[0, 0], [423, 9], [422, 52], [0, 41]]` → 文字: `BERT文本分类项目构建`
- 框[1]: `[[646, 24], [832, 24], [832, 44], [646, 44]]` → 文字: `Layer (type:depth-idx)`
- 框[2]: `[[331, 58], [545, 58], [545, 93], [331, 93]]` → 文字: `text \t label`
- ... 共 72 个

### `model_settings` — 模型设置

```json
{
  "use_doc_preprocessor": true,
  "use_textline_orientation": true
}
```

### `text_det_params` — 文本检测参数

```json
{
  "limit_side_len": 64,
  "limit_type": "min",
  "thresh": 0.3,
  "max_side_limit": 4000,
  "box_thresh": 0.6,
  "unclip_ratio": 1.5
}
```

### `text_type` — 文本类型

```
general
```

### `text_rec_score_thresh` — 识别分数阈值

```
0.0
```
> 低于此值的识别结果被丢弃。`0.0` 表示不过滤

### `return_word_box` — 是否返回词级包围框

```
False
```

### `rec_texts` / `rec_scores` / `rec_polys` — 识别三件套

共 `72` 条。前 10 条：

| # | rec_texts | rec_scores | rec_polys (左上角) |
|---|-----------|------------|---------------------|
| 0 | BERT文本分类项目构建 | 100.0% | (0,0) |
| 1 | Layer (type:depth-idx) | 99.7% | (646,24) |
| 2 | text \t label | 99.9% | (331,58) |
| 3 | BertModel | 99.8% | (644,59) |
| 4 | BertEmbeddings: 1-1 | 98.6% | (651,79) |
| 5 | 1.构建数据集 | 100.0% | (0,92) |
| 6 | CLS text | 99.7% | (254,106) |
| 7 | Embedding: 2-1 | 96.4% | (687,97) |
| 8 | SEP PAD | 96.1% | (434,113) |
| 9 | Embedding: 2-2 | 99.1% | (687,113) |
| ... | ... | ... | ... |

### `vis_fonts` — 可视化字体

共 `72` 个 Font 对象，用于绘制识别结果

### `textline_orientation_angles` — 文本行方向角

全 `72` 行，取值分布: {0: 72}

> `0`=正向 `90`=右旋 `180`=倒置 `270`=左旋

### `rec_boxes` — 包围盒 (x,y,w,h)

ndarray shape=`(72, 4)` dtype=`int16`

前 5 行 `[x, y, w, h]`：

| # | x | y | w | h | 对应文字 |
|---|---|---|---|---|----------|
| 0 | 0 | 0 | 423 | 52 | BERT文本分类项目构建 |
| 1 | 646 | 24 | 832 | 44 | Layer (type:depth-idx) |
| 2 | 331 | 58 | 545 | 93 | text \t label |
| 3 | 644 | 59 | 726 | 77 | BertModel |
| 4 | 650 | 79 | 824 | 98 | BertEmbeddings: 1-1 |

---
