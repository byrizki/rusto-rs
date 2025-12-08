import onnx
from onnx import NodeProto
import re
from tabulate import tabulate


# ---------------------------------------------------------------------------
# 1. Burn ONNX operator table (paste the table section exactly as-is)
# ---------------------------------------------------------------------------

OP_TABLE = """
| ONNX OP                          | Import Support | Burn Support |
|----------------------------------|:--------------:|:------------:|
| [Abs][1]                         | ✅             | ✅           |
| [Acos][2]                        | ❌             | ❌           |
| [Acosh][3]                       | ❌             | ❌           |
| [Add][4]                         | ✅             | ✅           |
| [AffineGrid][195]                | ❌             | ❌           |
| [And][5]                         | ✅             | ✅           |
| [ArgMax][6]                      | ✅             | ✅           |
| [ArgMin][7]                      | ✅             | ✅           |
| [Asin][8]                        | ❌             | ❌           |
| [Asinh][9]                       | ❌             | ❌           |
| [Atan][10]                       | ❌             | ❌           |
| [Atanh][11]                      | ❌             | ❌           |
| [Attention][194]                 | ✅             | ✅           |
| [AveragePool1d][12]              | ✅             | ✅           |
| [AveragePool2d][12]              | ✅             | ✅           |
| [BatchNormalization][14]         | ✅             | ✅           |
| [Bernoulli][15]                  | ✅             | ✅           |
| [BitShift][16]                   | ✅             | ✅           |
| [BitwiseAnd][17]                 | ✅             | ✅           |
| [BitwiseNot][18]                 | ✅             | ✅           |
| [BitwiseOr][19]                  | ✅             | ✅           |
| [BitwiseXor][20]                 | ✅             | ✅           |
| [BlackmanWindow][21]             | ❌             | ❌           |
| [Cast][22]                       | ✅             | ✅           |
| [CastLike][23]                   | ❌             | ❌           |
| [Ceil][24]                       | ✅             | ✅           |
| [Celu][25]                       | ❌             | ❌           |
| [CenterCropPad][26]              | ❌             | ❌           |
| [Clip][27]                       | ✅             | ✅           |
| [Col2Im][28]                     | ❌             | ❌           |
| [Compress][29]                   | ❌             | ❌           |
| [Concat][30]                     | ✅             | ✅           |
| [ConcatFromSequence][31]         | ❌             | ❌           |
| [Constant][32]                   | ✅             | ✅           |
| [ConstantOfShape][33]            | ✅             | ✅           |
| [Conv1d][34]                     | ✅             | ✅           |
| [Conv2d][34]                     | ✅             | ✅           |
| [Conv3d][34]                     | ✅             | ✅           |
| [ConvInteger][37]                | ❌             | ❌           |
| [ConvTranspose1d][38]            | ✅             | ✅           |
| [ConvTranspose2d][38]            | ✅             | ✅           |
| [ConvTranspose3d][38]            | ✅             | ✅           |
| [Cos][39]                        | ✅             | ✅           |
| [Cosh][40]                       | ✅             | ✅           |
| [CumSum][41]                     | ❌             | ✅          |
| [DeformConv][196]                | ❌             | ❌           |
| [DepthToSpace][42]               | ✅             | ✅           |
| [DequantizeLinear][43]           | ❌             | ❌           |
| [Det][44]                        | ❌             | ❌           |
| [DFT][45]                        | ❌             | ❌           |
| [Div][46]                        | ✅             | ✅           |
| [Dropout][47]                    | ✅             | ✅           |
| [DynamicQuantizeLinear][48]      | ❌             | ❌           |
| [Einsum][49]                     | ❌             | ❌           |
| [Elu][50]                        | ❌             | ❌           |
| [Equal][51]                      | ✅             | ✅           |
| [Erf][52]                        | ✅             | ✅           |
| [Exp][53]                        | ✅             | ✅           |
| [Expand][54]                     | ✅             | ✅           |
| [EyeLike][55]                    | ✅             | ✅           |
| [Flatten][56]                    | ✅             | ✅           |
| [Floor][57]                      | ✅             | ✅           |
| [Gather][58]                     | ✅             | ✅           |
| [GatherElements][59]             | ✅             | ✅           |
| [GatherND][60]                   | ❌             | ❌           |
| [Gelu][61]                       | ✅             | ✅           |
| [Gemm][62]                       | ✅             | ✅           |
| [GlobalAveragePool][63]          | ✅             | ✅           |
| [GlobalLpPool][64]               | ❌             | ❌           |
| [GlobalMaxPool][65]              | ❌             | ❌           |
| [Greater][66]                    | ✅             | ✅           |
| [GreaterOrEqual][67]             | ✅             | ✅           |
| [GridSample][68]                 | ✅             | ✅           |
| [GroupNormalization][69]         | ✅             | ✅           |
| [GRU][70]                        | ❌             | ✅           |
| [HammingWindow][71]              | ❌             | ❌           |
| [HannWindow][72]                 | ❌             | ❌           |
| [Hardmax][73]                    | ❌             | ❌           |
| [HardSigmoid][74]                | ✅             | ✅           |
| [HardSwish][75]                  | ❌             | ❌           |
| [Identity][76]                   | ✅             | ✅           |
| [If][77]                         | ❌             | ✅           |
| [Im][78]                         | ❌             | ❌           |
| [ImageDecoder][197]              | ❌             | ❌           |
| [InstanceNormalization][79]      | ✅             | ✅           |
| [IsInf][80]                      | ✅             | ✅           |
| [IsNaN][81]                      | ✅             | ✅           |
| [LayerNormalization][82]         | ✅             | ✅           |
| [LeakyRelu][83]                  | ✅             | ✅           |
| [Less][84]                       | ✅             | ✅           |
| [LessOrEqual][85]                | ✅             | ✅           |
| Linear                           | ✅             | ✅           |
| [Log][87]                        | ✅             | ✅           |
| [LogSoftmax][88]                 | ✅             | ✅           |
| [Loop][89]                       | ✅             | ✅           |
| [LpNormalization][90]            | ❌             | ❌           |
| [LpPool][91]                     | ❌             | ❌           |
| [LRN][92]                        | ❌             | ❌           |
| [LSTM][93]                       | ✅             | ✅           |
| [MatMul][94]                     | ✅             | ✅           |
| [MatMulInteger][95]              | ✅             | ✅           |
| [Max][96]                        | ✅             | ✅           |
| [MaxPool1d][97]                  | ✅             | ✅           |
| [MaxPool2d][98]                  | ✅             | ✅           |
| [MaxRoiPool][99]                 | ❌             | ❌           |
| [MaxUnpool][100]                 | ❌             | ❌           |
| [Mean][101]                      | ✅             | ✅           |
| [MeanVarianceNormalization][102] | ❌             | ❌           |
| [MelWeightMatrix][103]           | ❌             | ❌           |
| [Min][104]                       | ✅             | ✅           |
| [Mish][105]                      | ❌             | ❌           |
| [Mod][106]                       | ✅             | ✅           |
| [Mul][107]                       | ✅             | ✅           |
| [Multinomial][108]               | ❌             | ❌           |
| [Neg][109]                       | ✅             | ✅           |
| [NegativeLogLikelihoodLoss][110] | ❌             | ❌           |
| [NonMaxSuppression][112]         | ❌             | ❌           |
| [NonZero][113]                   | ✅             | ✅           |
| [Not][114]                       | ✅             | ✅           |
| [OneHot][115]                    | ✅             | ✅           |
| [Optional][116]                  | ❌             | ❌           |
| [OptionalGetElement][117]        | ❌             | ❌           |
| [OptionalHasElement][118]        | ❌             | ❌           |
| [Or][119]                        | ✅             | ✅           |
| [Pad][120]                       | ✅             | ✅           |
| [Pow][121]                       | ✅             | ✅           |
| [PRelu][122]                     | ✅             | ✅           |
| [QLinearConv][123]               | ❌             | ❌           |
| [QLinearMatMul][124]             | ❌             | ❌           |
| [QuantizeLinear][125]            | ❌             | ❌           |
| [RMSNormalization][198]          | ❌             | ❌           |
| [RNN][145]                       | ❌             | ✅           |
| [RandomNormal][126]              | ✅             | ✅           |
| [RandomNormalLike][127]          | ✅             | ✅           |
| [RandomUniform][128]             | ✅             | ✅           |
| [RandomUniformLike][129]         | ✅             | ✅           |
| [Range][130]                     | ✅             | ✅           |
| [Reciprocal][131]                | ✅             | ✅           |
| [ReduceL][132]                   | ✅             | ✅           |
| [ReduceLogSum][133]              | ✅             | ✅           |
| [ReduceLogSumExp][134]           | ✅             | ✅           |
| [ReduceMax][135]                 | ✅             | ✅           |
| [ReduceMean][136]                | ✅             | ✅           |
| [ReduceMin][137]                 | ✅             | ✅           |
| [ReduceProd][138]                | ✅             | ✅           |
| [ReduceSum][139]                 | ✅             | ✅           |
| [ReduceSumSquare][140]           | ✅             | ✅           |
| [RegexFullMatch][199]            | ❌             | ❌           |
| [Relu][141]                      | ✅             | ✅           |
| [Reshape][142]                   | ✅             | ✅           |
| [Resize][143]                    | ✅             | ✅           |
| [ReverseSequence][144]           | ❌             | ❌           |
| [RoiAlign][146]                  | ❌             | ❌           |
| [RotaryEmbedding][200]           | ❌             | ❌           |
| [Round][147]                     | ✅             | ✅           |
| [Scan][148]                      | ✅             | ✅           |
| [Scatter][149]                   | ❌             | ✅           |
| [ScatterElements][150]           | ❌             | ❌           |
| [ScatterND][151]                 | ❌             | ❌           |
| [Selu][152]                      | ❌             | ❌           |
| [SequenceAt][153]                | ❌             | ❌           |
| [SequenceConstruct][154]         | ❌             | ❌           |
| [SequenceEmpty][155]             | ❌             | ❌           |
| [SequenceErase][156]             | ❌             | ❌           |
| [SequenceInsert][157]            | ❌             | ❌           |
| [SequenceLength][158]            | ❌             | ❌           |
| [SequenceMap][159]               | ❌             | ❌           |
| [Shape][160]                     | ✅             | ✅           |
| [Shrink][161]                    | ❌             | ❌           |
| [Sigmoid][162]                   | ✅             | ✅           |
| [Sign][163]                      | ✅             | ✅           |
| [Sin][164]                       | ✅             | ✅           |
| [Sinh][165]                      | ✅             | ✅           |
| [Size][166]                      | ✅             | ✅           |
| [Slice][167]                     | ✅             | ✅           |
| [Softmax][168]                   | ✅             | ✅           |
| [SoftmaxCrossEntropyLoss][169]   | ❌             | ❌           |
| [Softplus][170]                  | ❌             | ❌           |
| [Softsign][171]                  | ❌             | ❌           |
| [SpaceToDepth][172]              | ✅             | ✅           |
| [Split][173]                     | ✅             | ✅           |
| [SplitToSequence][174]           | ❌             | ❌           |
| [Sqrt][175]                      | ✅             | ✅           |
| [Squeeze][176]                   | ✅             | ✅           |
| [STFT][177]                      | ❌             | ❌           |
| [StringConcat][201]              | ❌             | ❌           |
| [StringNormalizer][178]          | ❌             | ❌           |
| [StringSplit][202]               | ❌             | ❌           |
| [Sub][179]                       | ✅             | ✅           |
| [Sum][180]                       | ✅             | ✅           |
| [Swish][203]                     | ❌             | ❌           |
| [Tan][181]                       | ✅             | ✅           |
| [Tanh][182]                      | ✅             | ✅           |
| [TensorScatter][204]             | ❌             | ❌           |
| [TfIdfVectorizer][183]           | ❌             | ❌           |
| [ThresholdedRelu][184]           | ❌             | ❌           |
| [Tile][185]                      | ✅             | ✅           |
| [TopK][186]                      | ✅             | ✅           |
| [Transpose][187]                 | ✅             | ✅           |
| [Trilu][188]                     | ✅             | ✅           |
| [Unique][189]                    | ❌             | ❌           |
| [Upsample][190]                  | ❌             | ❌           |
| [Where][191]                     | ✅             | ✅           |
| [Xor][192]                       | ✅             | ✅           |
| [Unsqueeze][193]                 | ✅             | ✅           |
"""

# ---------------------------------------------------------------------------
# 2. Parse table into a Python dictionary
# ---------------------------------------------------------------------------

def parse_table(table_text):
    ops = {}
    for line in table_text.splitlines():
        if not line.startswith("| "):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) != 3 or cols[0] == "ONNX OP":
            continue

        opname, import_s, burn_s = cols

        # Extract plain operator name from markdown like "[Add][4]" -> "Add".
        m = re.search(r"\[([^\]]+)\]", opname)
        key = m.group(1) if m else opname.split()[0]

        ops[key] = {
            "import": import_s == "✅",
            "burn": burn_s == "✅"
        }
    return ops

burn_ops = parse_table(OP_TABLE)

# ---------------------------------------------------------------------------
# 3. Extract operator types used in ONNX model
# ---------------------------------------------------------------------------

def extract_ops_from_model(path):
    model = onnx.load(path)
    used_ops = set()
    for node in model.graph.node:
        used_ops.add(node.op_type)
    return sorted(list(used_ops))


# ---------------------------------------------------------------------------
# 4. Compare ONNX model ops with Burn compatibility table
# ---------------------------------------------------------------------------

def check_compatibility(model_ops):
    report = []
    missing = []

    for op in model_ops:
        if op in burn_ops:
            report.append([
                op,
                "✅" if burn_ops[op]["import"] else "❌",
                "✅" if burn_ops[op]["burn"] else "❌"
            ])
        else:
            missing.append(op)
            report.append([op, "❓", "❓"])

    return report, missing


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main(onnx_path):
    print(f"\n=== Checking ONNX model: {onnx_path} ===\n")

    model_ops = extract_ops_from_model(onnx_path)
    report, missing = check_compatibility(model_ops)

    print(tabulate(report, headers=["Operator", "Import", "Burn"], tablefmt="github"))

    if missing:
        print("\n⚠ Missing from Burn table:")
        for m in missing:
            print(f"  - {m}")

    unsupported = [r[0] for r in report if r[2] == "❌"]
    if unsupported:
        print("\n❌ Burn does NOT support these operators:")
        for u in unsupported:
            print(f"  - {u}")

    print("\nDone.\n")


if __name__ == "__main__":
    # Change ONNX file path
    main("models/PPOCR_v5/det.onnx")
