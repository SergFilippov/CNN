
D
PlaceholderPlaceholder*
dtype0*
shape:@
:
Placeholder_1Placeholder*
dtype0	*
shape:@
S
truncated_normal/shapeConst*
dtype0*%
valueB"             
B
truncated_normal/meanConst*
dtype0*
valueB
 *    
D
truncated_normal/stddevConst*
dtype0*
valueB
 *ЭЬЬ=
z
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
seed2 *

seed *
T0
_
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0
M
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0
b
VariableVariable*
dtype0*
shape: *
	container *
shared_name 
g
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
use_locking(*
T0
,
Variable/readIdentityVariable*
T0
6
zerosConst*
dtype0*
valueB *    
X

Variable_1Variable*
dtype0*
shape: *
	container *
shared_name 
`
Variable_1/AssignAssign
Variable_1zeros*
validate_shape(*
use_locking(*
T0
0
Variable_1/readIdentity
Variable_1*
T0
U
truncated_normal_1/shapeConst*
dtype0*%
valueB"          @   
D
truncated_normal_1/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_1/stddevConst*
dtype0*
valueB
 *ЭЬЬ=
~
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
seed2 *

seed *
T0
e
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0
S
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0
d

Variable_2Variable*
dtype0*
shape: @*
	container *
shared_name 
m
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
use_locking(*
T0
0
Variable_2/readIdentity
Variable_2*
T0
6
ConstConst*
dtype0*
valueB@*ЭЬЬ=
X

Variable_3Variable*
dtype0*
shape:@*
	container *
shared_name 
`
Variable_3/AssignAssign
Variable_3Const*
validate_shape(*
use_locking(*
T0
0
Variable_3/readIdentity
Variable_3*
T0
M
truncated_normal_2/shapeConst*
dtype0*
valueB"@     
D
truncated_normal_2/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_2/stddevConst*
dtype0*
valueB
 *ЭЬЬ=
~
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
seed2 *

seed *
T0
e
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0
S
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0
^

Variable_4Variable*
dtype0*
shape:
Р*
	container *
shared_name 
m
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
use_locking(*
T0
0
Variable_4/readIdentity
Variable_4*
T0
9
Const_1Const*
dtype0*
valueB*ЭЬЬ=
Y

Variable_5Variable*
dtype0*
shape:*
	container *
shared_name 
b
Variable_5/AssignAssign
Variable_5Const_1*
validate_shape(*
use_locking(*
T0
0
Variable_5/readIdentity
Variable_5*
T0
M
truncated_normal_3/shapeConst*
dtype0*
valueB"   
   
D
truncated_normal_3/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_3/stddevConst*
dtype0*
valueB
 *ЭЬЬ=
~
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
seed2 *

seed *
T0
e
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0
S
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0
]

Variable_6Variable*
dtype0*
shape:	
*
	container *
shared_name 
m
Variable_6/AssignAssign
Variable_6truncated_normal_3*
validate_shape(*
use_locking(*
T0
0
Variable_6/readIdentity
Variable_6*
T0
8
Const_2Const*
dtype0*
valueB
*ЭЬЬ=
X

Variable_7Variable*
dtype0*
shape:
*
	container *
shared_name 
b
Variable_7/AssignAssign
Variable_7Const_2*
validate_shape(*
use_locking(*
T0
0
Variable_7/readIdentity
Variable_7*
T0
u
Conv2DConv2DPlaceholderVariable/read*
paddingSAME*
strides
*
T0*
use_cudnn_on_gpu(
4
BiasAddBiasAddConv2DVariable_1/read*
T0

ReluReluBiasAdd*
T0
S
MaxPoolMaxPoolRelu*
paddingSAME*
strides
*
ksize

u
Conv2D_1Conv2DMaxPoolVariable_2/read*
paddingSAME*
strides
*
T0*
use_cudnn_on_gpu(
8
	BiasAdd_1BiasAddConv2D_1Variable_3/read*
T0
"
Relu_1Relu	BiasAdd_1*
T0
W
	MaxPool_1MaxPoolRelu_1*
paddingSAME*
strides
*
ksize

B
Reshape/shapeConst*
dtype0*
valueB"@   @  
5
ReshapeReshape	MaxPool_1Reshape/shape*
T0
Y
MatMulMatMulReshapeVariable_4/read*
transpose_b( *
transpose_a( *
T0
,
addAddMatMulVariable_5/read*
T0

Relu_2Reluadd*
T0
>
dropout/keep_probConst*
dtype0*
valueB
 *   ?
'
dropout/ShapeShapeRelu_2*
T0
G
dropout/random_uniform/minConst*
dtype0*
valueB
 *    
G
dropout/random_uniform/maxConst*
dtype0*
valueB
 *  ?
s
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
dtype0*
seed2 *

seed *
T0
b
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0
l
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0
^
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0
F
dropout/addAdddropout/keep_probdropout/random_uniform*
T0
,
dropout/FloorFloordropout/add*
T0
.
dropout/InvInvdropout/keep_prob*
T0
0
dropout/mulMulRelu_2dropout/Inv*
T0
9
dropout/mul_1Muldropout/muldropout/Floor*
T0
a
MatMul_1MatMuldropout/mul_1Variable_6/read*
transpose_b( *
transpose_a( *
T0
0
add_1AddMatMul_1Variable_7/read*
T0
i
#SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsadd_1Placeholder_1*
T0
:
RankRank#SparseSoftmaxCrossEntropyWithLogits*
T0
5
range/startConst*
dtype0*
value	B : 
5
range/deltaConst*
dtype0*
value	B :
.
rangeRangerange/startRankrange/delta
R
MeanMean#SparseSoftmaxCrossEntropyWithLogitsrange*
T0*
	keep_dims( 
*
L2LossL2LossVariable_4/read*
T0
,
L2Loss_1L2LossVariable_5/read*
T0
'
add_2AddL2LossL2Loss_1*
T0
,
L2Loss_2L2LossVariable_6/read*
T0
&
add_3Addadd_2L2Loss_2*
T0
,
L2Loss_3L2LossVariable_7/read*
T0
&
add_4Addadd_3L2Loss_3*
T0
2
mul/xConst*
dtype0*
valueB
 *o:
!
mulMulmul/xadd_4*
T0
 
add_5AddMeanmul*
T0
B
Variable_8/initial_valueConst*
dtype0*
value	B : 
T

Variable_8Variable*
dtype0*
shape: *
	container *
shared_name 
s
Variable_8/AssignAssign
Variable_8Variable_8/initial_value*
validate_shape(*
use_locking(*
T0
0
Variable_8/readIdentity
Variable_8*
T0
1
mul_1/yConst*
dtype0*
value	B :@
/
mul_1MulVariable_8/readmul_1/y*
T0
K
ExponentialDecay/learning_rateConst*
dtype0*
valueB
 *
з#<
<
ExponentialDecay/CastCastmul_1*

DstT0*

SrcT0
E
ExponentialDecay/Cast_1/xConst*
dtype0*
valueB	 :рд
R
ExponentialDecay/Cast_1CastExponentialDecay/Cast_1/x*

DstT0*

SrcT0
F
ExponentialDecay/Cast_2/xConst*
dtype0*
valueB
 *33s?
X
ExponentialDecay/truedivDivExponentialDecay/CastExponentialDecay/Cast_1*
T0
B
ExponentialDecay/FloorFloorExponentialDecay/truediv*
T0
W
ExponentialDecay/PowPowExponentialDecay/Cast_2/xExponentialDecay/Floor*
T0
V
ExponentialDecayMulExponentialDecay/learning_rateExponentialDecay/Pow*
T0
(
gradients/ShapeShapeadd_5*
T0
<
gradients/ConstConst*
dtype0*
valueB
 *  ?
A
gradients/FillFillgradients/Shapegradients/Const*
T0
2
gradients/add_5_grad/ShapeShapeMean*
T0
3
gradients/add_5_grad/Shape_1Shapemul*
T0
}
*gradients/add_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_5_grad/Shapegradients/add_5_grad/Shape_1
u
gradients/add_5_grad/SumSumgradients/Fill*gradients/add_5_grad/BroadcastGradientArgs*
T0*
	keep_dims( 
f
gradients/add_5_grad/ReshapeReshapegradients/add_5_grad/Sumgradients/add_5_grad/Shape*
T0
y
gradients/add_5_grad/Sum_1Sumgradients/Fill,gradients/add_5_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( 
l
gradients/add_5_grad/Reshape_1Reshapegradients/add_5_grad/Sum_1gradients/add_5_grad/Shape_1*
T0
m
%gradients/add_5_grad/tuple/group_depsNoOp^gradients/add_5_grad/Reshape^gradients/add_5_grad/Reshape_1

-gradients/add_5_grad/tuple/control_dependencyIdentitygradients/add_5_grad/Reshape&^gradients/add_5_grad/tuple/group_deps*
T0

/gradients/add_5_grad/tuple/control_dependency_1Identitygradients/add_5_grad/Reshape_1&^gradients/add_5_grad/tuple/group_deps*
T0
P
gradients/Mean_grad/ShapeShape#SparseSoftmaxCrossEntropyWithLogits*
T0
N
gradients/Mean_grad/RankRank#SparseSoftmaxCrossEntropyWithLogits*
T0
4
gradients/Mean_grad/Shape_1Shaperange*
T0
I
gradients/Mean_grad/range/startConst*
dtype0*
value	B : 
I
gradients/Mean_grad/range/deltaConst*
dtype0*
value	B :
~
gradients/Mean_grad/rangeRangegradients/Mean_grad/range/startgradients/Mean_grad/Rankgradients/Mean_grad/range/delta
H
gradients/Mean_grad/Fill/valueConst*
dtype0*
value	B :
f
gradients/Mean_grad/FillFillgradients/Mean_grad/Shape_1gradients/Mean_grad/Fill/value*
T0

!gradients/Mean_grad/DynamicStitchDynamicStitchgradients/Mean_grad/rangerangegradients/Mean_grad/Shapegradients/Mean_grad/Fill*
T0*
N
j
gradients/Mean_grad/floordivDivgradients/Mean_grad/Shape!gradients/Mean_grad/DynamicStitch*
T0

gradients/Mean_grad/ReshapeReshape-gradients/add_5_grad/tuple/control_dependency!gradients/Mean_grad/DynamicStitch*
T0
d
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/floordiv*
T0
R
gradients/Mean_grad/Shape_2Shape#SparseSoftmaxCrossEntropyWithLogits*
T0
3
gradients/Mean_grad/Shape_3ShapeMean*
T0
H
gradients/Mean_grad/Rank_1Rankgradients/Mean_grad/Shape_2*
T0
K
!gradients/Mean_grad/range_1/startConst*
dtype0*
value	B : 
K
!gradients/Mean_grad/range_1/deltaConst*
dtype0*
value	B :

gradients/Mean_grad/range_1Range!gradients/Mean_grad/range_1/startgradients/Mean_grad/Rank_1!gradients/Mean_grad/range_1/delta
t
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_2gradients/Mean_grad/range_1*
T0*
	keep_dims( 
H
gradients/Mean_grad/Rank_2Rankgradients/Mean_grad/Shape_3*
T0
K
!gradients/Mean_grad/range_2/startConst*
dtype0*
value	B : 
K
!gradients/Mean_grad/range_2/deltaConst*
dtype0*
value	B :

gradients/Mean_grad/range_2Range!gradients/Mean_grad/range_2/startgradients/Mean_grad/Rank_2!gradients/Mean_grad/range_2/delta
v
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_3gradients/Mean_grad/range_2*
T0*
	keep_dims( 
d
gradients/Mean_grad/floordiv_1Divgradients/Mean_grad/Prodgradients/Mean_grad/Prod_1*
T0
X
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv_1*

DstT0*

SrcT0
_
gradients/Mean_grad/truedivDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0
1
gradients/mul_grad/ShapeShapemul/x*
T0
3
gradients/mul_grad/Shape_1Shapeadd_4*
T0
w
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1
^
gradients/mul_grad/mulMul/gradients/add_5_grad/tuple/control_dependency_1add_4*
T0
y
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( 
`
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0
`
gradients/mul_grad/mul_1Mulmul/x/gradients/add_5_grad/tuple/control_dependency_1*
T0

gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( 
f
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1

+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0

-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0
3
gradients/add_4_grad/ShapeShapeadd_3*
T0
8
gradients/add_4_grad/Shape_1ShapeL2Loss_3*
T0
}
*gradients/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_4_grad/Shapegradients/add_4_grad/Shape_1

gradients/add_4_grad/SumSum-gradients/mul_grad/tuple/control_dependency_1*gradients/add_4_grad/BroadcastGradientArgs*
T0*
	keep_dims( 
f
gradients/add_4_grad/ReshapeReshapegradients/add_4_grad/Sumgradients/add_4_grad/Shape*
T0

gradients/add_4_grad/Sum_1Sum-gradients/mul_grad/tuple/control_dependency_1,gradients/add_4_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( 
l
gradients/add_4_grad/Reshape_1Reshapegradients/add_4_grad/Sum_1gradients/add_4_grad/Shape_1*
T0
m
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/add_4_grad/Reshape^gradients/add_4_grad/Reshape_1

-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/add_4_grad/Reshape&^gradients/add_4_grad/tuple/group_deps*
T0

/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/add_4_grad/Reshape_1&^gradients/add_4_grad/tuple/group_deps*
T0
3
gradients/add_3_grad/ShapeShapeadd_2*
T0
8
gradients/add_3_grad/Shape_1ShapeL2Loss_2*
T0
}
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1

gradients/add_3_grad/SumSum-gradients/add_4_grad/tuple/control_dependency*gradients/add_3_grad/BroadcastGradientArgs*
T0*
	keep_dims( 
f
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
T0

gradients/add_3_grad/Sum_1Sum-gradients/add_4_grad/tuple/control_dependency,gradients/add_3_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( 
l
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
T0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1

-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
T0

/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
T0
m
gradients/L2Loss_3_grad/mulMulVariable_7/read/gradients/add_4_grad/tuple/control_dependency_1*
T0
[
gradients/zeros_like/ZerosLike	ZerosLike%SparseSoftmaxCrossEntropyWithLogits:1*
T0
t
Agradients/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
valueB :
џџџџџџџџџ
Д
=gradients/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsgradients/Mean_grad/truedivAgradients/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0
Ќ
6gradients/SparseSoftmaxCrossEntropyWithLogits_grad/mulMul=gradients/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims%SparseSoftmaxCrossEntropyWithLogits:1*
T0
4
gradients/add_2_grad/ShapeShapeL2Loss*
T0
8
gradients/add_2_grad/Shape_1ShapeL2Loss_1*
T0
}
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1

gradients/add_2_grad/SumSum-gradients/add_3_grad/tuple/control_dependency*gradients/add_2_grad/BroadcastGradientArgs*
T0*
	keep_dims( 
f
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
T0

gradients/add_2_grad/Sum_1Sum-gradients/add_3_grad/tuple/control_dependency,gradients/add_2_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( 
l
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
T0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1

-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0

/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
T0
m
gradients/L2Loss_2_grad/mulMulVariable_6/read/gradients/add_3_grad/tuple/control_dependency_1*
T0
6
gradients/add_1_grad/ShapeShapeMatMul_1*
T0
?
gradients/add_1_grad/Shape_1ShapeVariable_7/read*
T0
}
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1

gradients/add_1_grad/SumSum6gradients/SparseSoftmaxCrossEntropyWithLogits_grad/mul*gradients/add_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( 
f
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
T0
Ё
gradients/add_1_grad/Sum_1Sum6gradients/SparseSoftmaxCrossEntropyWithLogits_grad/mul,gradients/add_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( 
l
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
T0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1

-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0

/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0
i
gradients/L2Loss_grad/mulMulVariable_4/read-gradients/add_2_grad/tuple/control_dependency*
T0
m
gradients/L2Loss_1_grad/mulMulVariable_5/read/gradients/add_2_grad/tuple/control_dependency_1*
T0

gradients/MatMul_1_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependencyVariable_6/read*
transpose_b(*
transpose_a( *
T0

 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul_1-gradients/add_1_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1

0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0

2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0
v
gradients/AddNAddNgradients/L2Loss_3_grad/mul/gradients/add_1_grad/tuple/control_dependency_1*
T0*
N
A
"gradients/dropout/mul_1_grad/ShapeShapedropout/mul*
T0
E
$gradients/dropout/mul_1_grad/Shape_1Shapedropout/Floor*
T0

2gradients/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout/mul_1_grad/Shape$gradients/dropout/mul_1_grad/Shape_1
q
 gradients/dropout/mul_1_grad/mulMul0gradients/MatMul_1_grad/tuple/control_dependencydropout/Floor*
T0

 gradients/dropout/mul_1_grad/SumSum gradients/dropout/mul_1_grad/mul2gradients/dropout/mul_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( 
~
$gradients/dropout/mul_1_grad/ReshapeReshape gradients/dropout/mul_1_grad/Sum"gradients/dropout/mul_1_grad/Shape*
T0
q
"gradients/dropout/mul_1_grad/mul_1Muldropout/mul0gradients/MatMul_1_grad/tuple/control_dependency*
T0

"gradients/dropout/mul_1_grad/Sum_1Sum"gradients/dropout/mul_1_grad/mul_14gradients/dropout/mul_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( 

&gradients/dropout/mul_1_grad/Reshape_1Reshape"gradients/dropout/mul_1_grad/Sum_1$gradients/dropout/mul_1_grad/Shape_1*
T0

-gradients/dropout/mul_1_grad/tuple/group_depsNoOp%^gradients/dropout/mul_1_grad/Reshape'^gradients/dropout/mul_1_grad/Reshape_1
 
5gradients/dropout/mul_1_grad/tuple/control_dependencyIdentity$gradients/dropout/mul_1_grad/Reshape.^gradients/dropout/mul_1_grad/tuple/group_deps*
T0
Є
7gradients/dropout/mul_1_grad/tuple/control_dependency_1Identity&gradients/dropout/mul_1_grad/Reshape_1.^gradients/dropout/mul_1_grad/tuple/group_deps*
T0
{
gradients/AddN_1AddNgradients/L2Loss_2_grad/mul2gradients/MatMul_1_grad/tuple/control_dependency_1*
T0*
N
:
 gradients/dropout/mul_grad/ShapeShapeRelu_2*
T0
A
"gradients/dropout/mul_grad/Shape_1Shapedropout/Inv*
T0

0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1
r
gradients/dropout/mul_grad/mulMul5gradients/dropout/mul_1_grad/tuple/control_dependencydropout/Inv*
T0

gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( 
x
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
T0
o
 gradients/dropout/mul_grad/mul_1MulRelu_25gradients/dropout/mul_1_grad/tuple/control_dependency*
T0

 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( 
~
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
T0

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1

3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
T0

5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
T0
p
gradients/Relu_2_grad/ReluGradReluGrad3gradients/dropout/mul_grad/tuple/control_dependencyRelu_2*
T0
2
gradients/add_grad/ShapeShapeMatMul*
T0
=
gradients/add_grad/Shape_1ShapeVariable_5/read*
T0
w
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1

gradients/add_grad/SumSumgradients/Relu_2_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
T0*
	keep_dims( 
`
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0

gradients/add_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( 
f
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1

+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0

-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0

gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_4/read*
transpose_b(*
transpose_a( *
T0

gradients/MatMul_grad/MatMul_1MatMulReshape+gradients/add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1

.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0

0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0
v
gradients/AddN_2AddNgradients/L2Loss_1_grad/mul-gradients/add_grad/tuple/control_dependency_1*
T0*
N
9
gradients/Reshape_grad/ShapeShape	MaxPool_1*
T0

gradients/Reshape_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*
T0
w
gradients/AddN_3AddNgradients/L2Loss_grad/mul0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
N
Ё
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1gradients/Reshape_grad/Reshape*
paddingSAME*
strides
*
ksize

a
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0
N
gradients/BiasAdd_1_grad/RankRankgradients/Relu_1_grad/ReluGrad*
T0
H
gradients/BiasAdd_1_grad/sub/yConst*
dtype0*
value	B :
k
gradients/BiasAdd_1_grad/subSubgradients/BiasAdd_1_grad/Rankgradients/BiasAdd_1_grad/sub/y*
T0
N
$gradients/BiasAdd_1_grad/range/startConst*
dtype0*
value	B : 
N
$gradients/BiasAdd_1_grad/range/deltaConst*
dtype0*
value	B :

gradients/BiasAdd_1_grad/rangeRange$gradients/BiasAdd_1_grad/range/startgradients/BiasAdd_1_grad/sub$gradients/BiasAdd_1_grad/range/delta
}
gradients/BiasAdd_1_grad/SumSumgradients/Relu_1_grad/ReluGradgradients/BiasAdd_1_grad/range*
T0*
	keep_dims( 
q
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Relu_1_grad/ReluGrad^gradients/BiasAdd_1_grad/Sum

1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0

3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identitygradients/BiasAdd_1_grad/Sum*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0
8
gradients/Conv2D_1_grad/ShapeShapeMaxPool*
T0
ю
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeVariable_2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*
paddingSAME*
strides
*
T0*
use_cudnn_on_gpu(
B
gradients/Conv2D_1_grad/Shape_1ShapeVariable_2/read*
T0
ъ
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPoolgradients/Conv2D_1_grad/Shape_11gradients/BiasAdd_1_grad/tuple/control_dependency*
paddingSAME*
strides
*
T0*
use_cudnn_on_gpu(

(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter

0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0
 
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0
­
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool0gradients/Conv2D_1_grad/tuple/control_dependency*
paddingSAME*
strides
*
ksize

[
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0
J
gradients/BiasAdd_grad/RankRankgradients/Relu_grad/ReluGrad*
T0
F
gradients/BiasAdd_grad/sub/yConst*
dtype0*
value	B :
e
gradients/BiasAdd_grad/subSubgradients/BiasAdd_grad/Rankgradients/BiasAdd_grad/sub/y*
T0
L
"gradients/BiasAdd_grad/range/startConst*
dtype0*
value	B : 
L
"gradients/BiasAdd_grad/range/deltaConst*
dtype0*
value	B :

gradients/BiasAdd_grad/rangeRange"gradients/BiasAdd_grad/range/startgradients/BiasAdd_grad/sub"gradients/BiasAdd_grad/range/delta
w
gradients/BiasAdd_grad/SumSumgradients/Relu_grad/ReluGradgradients/BiasAdd_grad/range*
T0*
	keep_dims( 
k
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad^gradients/BiasAdd_grad/Sum

/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0

1gradients/BiasAdd_grad/tuple/control_dependency_1Identitygradients/BiasAdd_grad/Sum(^gradients/BiasAdd_grad/tuple/group_deps*
T0
:
gradients/Conv2D_grad/ShapeShapePlaceholder*
T0
ц
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeVariable/read/gradients/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
strides
*
T0*
use_cudnn_on_gpu(
>
gradients/Conv2D_grad/Shape_1ShapeVariable/read*
T0
ш
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholdergradients/Conv2D_grad/Shape_1/gradients/BiasAdd_grad/tuple/control_dependency*
paddingSAME*
strides
*
T0*
use_cudnn_on_gpu(

&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter

.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0

0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0
D
zeros_1Const*
dtype0*%
valueB *    
k
Variable/MomentumVariable*
dtype0*
shape: *
	container *
shared_name 
p
Variable/Momentum/AssignAssignVariable/Momentumzeros_1*
validate_shape(*
use_locking(*
T0
>
Variable/Momentum/readIdentityVariable/Momentum*
T0
8
zeros_2Const*
dtype0*
valueB *    
a
Variable_1/MomentumVariable*
dtype0*
shape: *
	container *
shared_name 
t
Variable_1/Momentum/AssignAssignVariable_1/Momentumzeros_2*
validate_shape(*
use_locking(*
T0
B
Variable_1/Momentum/readIdentityVariable_1/Momentum*
T0
D
zeros_3Const*
dtype0*%
valueB @*    
m
Variable_2/MomentumVariable*
dtype0*
shape: @*
	container *
shared_name 
t
Variable_2/Momentum/AssignAssignVariable_2/Momentumzeros_3*
validate_shape(*
use_locking(*
T0
B
Variable_2/Momentum/readIdentityVariable_2/Momentum*
T0
8
zeros_4Const*
dtype0*
valueB@*    
a
Variable_3/MomentumVariable*
dtype0*
shape:@*
	container *
shared_name 
t
Variable_3/Momentum/AssignAssignVariable_3/Momentumzeros_4*
validate_shape(*
use_locking(*
T0
B
Variable_3/Momentum/readIdentityVariable_3/Momentum*
T0
>
zeros_5Const*
dtype0*
valueB
Р*    
g
Variable_4/MomentumVariable*
dtype0*
shape:
Р*
	container *
shared_name 
t
Variable_4/Momentum/AssignAssignVariable_4/Momentumzeros_5*
validate_shape(*
use_locking(*
T0
B
Variable_4/Momentum/readIdentityVariable_4/Momentum*
T0
9
zeros_6Const*
dtype0*
valueB*    
b
Variable_5/MomentumVariable*
dtype0*
shape:*
	container *
shared_name 
t
Variable_5/Momentum/AssignAssignVariable_5/Momentumzeros_6*
validate_shape(*
use_locking(*
T0
B
Variable_5/Momentum/readIdentityVariable_5/Momentum*
T0
=
zeros_7Const*
dtype0*
valueB	
*    
f
Variable_6/MomentumVariable*
dtype0*
shape:	
*
	container *
shared_name 
t
Variable_6/Momentum/AssignAssignVariable_6/Momentumzeros_7*
validate_shape(*
use_locking(*
T0
B
Variable_6/Momentum/readIdentityVariable_6/Momentum*
T0
8
zeros_8Const*
dtype0*
valueB
*    
a
Variable_7/MomentumVariable*
dtype0*
shape:
*
	container *
shared_name 
t
Variable_7/Momentum/AssignAssignVariable_7/Momentumzeros_8*
validate_shape(*
use_locking(*
T0
B
Variable_7/Momentum/readIdentityVariable_7/Momentum*
T0
>
Momentum/momentumConst*
dtype0*
valueB
 *fff?
Ч
&Momentum/update_Variable/ApplyMomentumApplyMomentumVariableVariable/MomentumExponentialDecay0gradients/Conv2D_grad/tuple/control_dependency_1Momentum/momentum*
use_locking( *
T0
Ю
(Momentum/update_Variable_1/ApplyMomentumApplyMomentum
Variable_1Variable_1/MomentumExponentialDecay1gradients/BiasAdd_grad/tuple/control_dependency_1Momentum/momentum*
use_locking( *
T0
Я
(Momentum/update_Variable_2/ApplyMomentumApplyMomentum
Variable_2Variable_2/MomentumExponentialDecay2gradients/Conv2D_1_grad/tuple/control_dependency_1Momentum/momentum*
use_locking( *
T0
а
(Momentum/update_Variable_3/ApplyMomentumApplyMomentum
Variable_3Variable_3/MomentumExponentialDecay3gradients/BiasAdd_1_grad/tuple/control_dependency_1Momentum/momentum*
use_locking( *
T0
­
(Momentum/update_Variable_4/ApplyMomentumApplyMomentum
Variable_4Variable_4/MomentumExponentialDecaygradients/AddN_3Momentum/momentum*
use_locking( *
T0
­
(Momentum/update_Variable_5/ApplyMomentumApplyMomentum
Variable_5Variable_5/MomentumExponentialDecaygradients/AddN_2Momentum/momentum*
use_locking( *
T0
­
(Momentum/update_Variable_6/ApplyMomentumApplyMomentum
Variable_6Variable_6/MomentumExponentialDecaygradients/AddN_1Momentum/momentum*
use_locking( *
T0
Ћ
(Momentum/update_Variable_7/ApplyMomentumApplyMomentum
Variable_7Variable_7/MomentumExponentialDecaygradients/AddNMomentum/momentum*
use_locking( *
T0
э
Momentum/updateNoOp'^Momentum/update_Variable/ApplyMomentum)^Momentum/update_Variable_1/ApplyMomentum)^Momentum/update_Variable_2/ApplyMomentum)^Momentum/update_Variable_3/ApplyMomentum)^Momentum/update_Variable_4/ApplyMomentum)^Momentum/update_Variable_5/ApplyMomentum)^Momentum/update_Variable_6/ApplyMomentum)^Momentum/update_Variable_7/ApplyMomentum
J
Momentum/valueConst^Momentum/update*
dtype0*
value	B :
M
Momentum	AssignAdd
Variable_8Momentum/value*
use_locking( *
T0
"
SoftmaxSoftmaxadd_1*
T0
Є
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable/Momentum/Assign^Variable_1/Momentum/Assign^Variable_2/Momentum/Assign^Variable_3/Momentum/Assign^Variable_4/Momentum/Assign^Variable_5/Momentum/Assign^Variable_6/Momentum/Assign^Variable_7/Momentum/Assign
8

save/ConstConst*
dtype0*
valueB Bmodel
д
save/save/tensor_namesConst*
dtype0*Ѕ
valueBBVariableBVariable/MomentumB
Variable_1BVariable_1/MomentumB
Variable_2BVariable_2/MomentumB
Variable_3BVariable_3/MomentumB
Variable_4BVariable_4/MomentumB
Variable_5BVariable_5/MomentumB
Variable_6BVariable_6/MomentumB
Variable_7BVariable_7/MomentumB
Variable_8
h
save/save/shapes_and_slicesConst*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 

	save/save
SaveSlices
save/Constsave/save/tensor_namessave/save/shapes_and_slicesVariableVariable/Momentum
Variable_1Variable_1/Momentum
Variable_2Variable_2/Momentum
Variable_3Variable_3/Momentum
Variable_4Variable_4/Momentum
Variable_5Variable_5/Momentum
Variable_6Variable_6/Momentum
Variable_7Variable_7/Momentum
Variable_8*
T
2
D
save/control_dependencyIdentity
save/Const
^save/save*
T0
O
save/restore_slice/tensor_nameConst*
dtype0*
valueB BVariable
K
"save/restore_slice/shape_and_sliceConst*
dtype0*
valueB B 

save/restore_sliceRestoreSlice
save/Constsave/restore_slice/tensor_name"save/restore_slice/shape_and_slice*
preferred_shardџџџџџџџџџ*
dt0
e
save/AssignAssignVariablesave/restore_slice*
validate_shape(*
use_locking(*
T0
Z
 save/restore_slice_1/tensor_nameConst*
dtype0*"
valueB BVariable/Momentum
M
$save/restore_slice_1/shape_and_sliceConst*
dtype0*
valueB B 
Ђ
save/restore_slice_1RestoreSlice
save/Const save/restore_slice_1/tensor_name$save/restore_slice_1/shape_and_slice*
preferred_shardџџџџџџџџџ*
dt0
r
save/Assign_1AssignVariable/Momentumsave/restore_slice_1*
validate_shape(*
use_locking(*
T0
S
 save/restore_slice_2/tensor_nameConst*
dtype0*
valueB B
Variable_1
M
$save/restore_slice_2/shape_and_sliceConst*
dtype0*
valueB B 
Ђ
save/restore_slice_2RestoreSlice
save/Const save/restore_slice_2/tensor_name$save/restore_slice_2/shape_and_slice*
preferred_shardџџџџџџџџџ*
dt0
k
save/Assign_2Assign
Variable_1save/restore_slice_2*
validate_shape(*
use_locking(*
T0
\
 save/restore_slice_3/tensor_nameConst*
dtype0*$
valueB BVariable_1/Momentum
M
$save/restore_slice_3/shape_and_sliceConst*
dtype0*
valueB B 
Ђ
save/restore_slice_3RestoreSlice
save/Const save/restore_slice_3/tensor_name$save/restore_slice_3/shape_and_slice*
preferred_shardџџџџџџџџџ*
dt0
t
save/Assign_3AssignVariable_1/Momentumsave/restore_slice_3*
validate_shape(*
use_locking(*
T0
S
 save/restore_slice_4/tensor_nameConst*
dtype0*
valueB B
Variable_2
M
$save/restore_slice_4/shape_and_sliceConst*
dtype0*
valueB B 
Ђ
save/restore_slice_4RestoreSlice
save/Const save/restore_slice_4/tensor_name$save/restore_slice_4/shape_and_slice*
preferred_shardџџџџџџџџџ*
dt0
k
save/Assign_4Assign
Variable_2save/restore_slice_4*
validate_shape(*
use_locking(*
T0
\
 save/restore_slice_5/tensor_nameConst*
dtype0*$
valueB BVariable_2/Momentum
M
$save/restore_slice_5/shape_and_sliceConst*
dtype0*
valueB B 
Ђ
save/restore_slice_5RestoreSlice
save/Const save/restore_slice_5/tensor_name$save/restore_slice_5/shape_and_slice*
preferred_shardџџџџџџџџџ*
dt0
t
save/Assign_5AssignVariable_2/Momentumsave/restore_slice_5*
validate_shape(*
use_locking(*
T0
S
 save/restore_slice_6/tensor_nameConst*
dtype0*
valueB B
Variable_3
M
$save/restore_slice_6/shape_and_sliceConst*
dtype0*
valueB B 
Ђ
save/restore_slice_6RestoreSlice
save/Const save/restore_slice_6/tensor_name$save/restore_slice_6/shape_and_slice*
preferred_shardџџџџџџџџџ*
dt0
k
save/Assign_6Assign
Variable_3save/restore_slice_6*
validate_shape(*
use_locking(*
T0
\
 save/restore_slice_7/tensor_nameConst*
dtype0*$
valueB BVariable_3/Momentum
M
$save/restore_slice_7/shape_and_sliceConst*
dtype0*
valueB B 
Ђ
save/restore_slice_7RestoreSlice
save/Const save/restore_slice_7/tensor_name$save/restore_slice_7/shape_and_slice*
preferred_shardџџџџџџџџџ*
dt0
t
save/Assign_7AssignVariable_3/Momentumsave/restore_slice_7*
validate_shape(*
use_locking(*
T0
S
 save/restore_slice_8/tensor_nameConst*
dtype0*
valueB B
Variable_4
M
$save/restore_slice_8/shape_and_sliceConst*
dtype0*
valueB B 
Ђ
save/restore_slice_8RestoreSlice
save/Const save/restore_slice_8/tensor_name$save/restore_slice_8/shape_and_slice*
preferred_shardџџџџџџџџџ*
dt0
k
save/Assign_8Assign
Variable_4save/restore_slice_8*
validate_shape(*
use_locking(*
T0
\
 save/restore_slice_9/tensor_nameConst*
dtype0*$
valueB BVariable_4/Momentum
M
$save/restore_slice_9/shape_and_sliceConst*
dtype0*
valueB B 
Ђ
save/restore_slice_9RestoreSlice
save/Const save/restore_slice_9/tensor_name$save/restore_slice_9/shape_and_slice*
preferred_shardџџџџџџџџџ*
dt0
t
save/Assign_9AssignVariable_4/Momentumsave/restore_slice_9*
validate_shape(*
use_locking(*
T0
T
!save/restore_slice_10/tensor_nameConst*
dtype0*
valueB B
Variable_5
N
%save/restore_slice_10/shape_and_sliceConst*
dtype0*
valueB B 
Ѕ
save/restore_slice_10RestoreSlice
save/Const!save/restore_slice_10/tensor_name%save/restore_slice_10/shape_and_slice*
preferred_shardџџџџџџџџџ*
dt0
m
save/Assign_10Assign
Variable_5save/restore_slice_10*
validate_shape(*
use_locking(*
T0
]
!save/restore_slice_11/tensor_nameConst*
dtype0*$
valueB BVariable_5/Momentum
N
%save/restore_slice_11/shape_and_sliceConst*
dtype0*
valueB B 
Ѕ
save/restore_slice_11RestoreSlice
save/Const!save/restore_slice_11/tensor_name%save/restore_slice_11/shape_and_slice*
preferred_shardџџџџџџџџџ*
dt0
v
save/Assign_11AssignVariable_5/Momentumsave/restore_slice_11*
validate_shape(*
use_locking(*
T0
T
!save/restore_slice_12/tensor_nameConst*
dtype0*
valueB B
Variable_6
N
%save/restore_slice_12/shape_and_sliceConst*
dtype0*
valueB B 
Ѕ
save/restore_slice_12RestoreSlice
save/Const!save/restore_slice_12/tensor_name%save/restore_slice_12/shape_and_slice*
preferred_shardџџџџџџџџџ*
dt0
m
save/Assign_12Assign
Variable_6save/restore_slice_12*
validate_shape(*
use_locking(*
T0
]
!save/restore_slice_13/tensor_nameConst*
dtype0*$
valueB BVariable_6/Momentum
N
%save/restore_slice_13/shape_and_sliceConst*
dtype0*
valueB B 
Ѕ
save/restore_slice_13RestoreSlice
save/Const!save/restore_slice_13/tensor_name%save/restore_slice_13/shape_and_slice*
preferred_shardџџџџџџџџџ*
dt0
v
save/Assign_13AssignVariable_6/Momentumsave/restore_slice_13*
validate_shape(*
use_locking(*
T0
T
!save/restore_slice_14/tensor_nameConst*
dtype0*
valueB B
Variable_7
N
%save/restore_slice_14/shape_and_sliceConst*
dtype0*
valueB B 
Ѕ
save/restore_slice_14RestoreSlice
save/Const!save/restore_slice_14/tensor_name%save/restore_slice_14/shape_and_slice*
preferred_shardџџџџџџџџџ*
dt0
m
save/Assign_14Assign
Variable_7save/restore_slice_14*
validate_shape(*
use_locking(*
T0
]
!save/restore_slice_15/tensor_nameConst*
dtype0*$
valueB BVariable_7/Momentum
N
%save/restore_slice_15/shape_and_sliceConst*
dtype0*
valueB B 
Ѕ
save/restore_slice_15RestoreSlice
save/Const!save/restore_slice_15/tensor_name%save/restore_slice_15/shape_and_slice*
preferred_shardџџџџџџџџџ*
dt0
v
save/Assign_15AssignVariable_7/Momentumsave/restore_slice_15*
validate_shape(*
use_locking(*
T0
T
!save/restore_slice_16/tensor_nameConst*
dtype0*
valueB B
Variable_8
N
%save/restore_slice_16/shape_and_sliceConst*
dtype0*
valueB B 
Ѕ
save/restore_slice_16RestoreSlice
save/Const!save/restore_slice_16/tensor_name%save/restore_slice_16/shape_and_slice*
preferred_shardџџџџџџџџџ*
dt0
m
save/Assign_16Assign
Variable_8save/restore_slice_16*
validate_shape(*
use_locking(*
T0
­
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16"