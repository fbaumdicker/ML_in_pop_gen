??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
}
dense_111/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'?*!
shared_namedense_111/kernel
v
$dense_111/kernel/Read/ReadVariableOpReadVariableOpdense_111/kernel*
_output_shapes
:	'?*
dtype0
u
dense_111/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_111/bias
n
"dense_111/bias/Read/ReadVariableOpReadVariableOpdense_111/bias*
_output_shapes	
:?*
dtype0
}
dense_112/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*!
shared_namedense_112/kernel
v
$dense_112/kernel/Read/ReadVariableOpReadVariableOpdense_112/kernel*
_output_shapes
:	?*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/dense_111/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'?*(
shared_nameAdam/dense_111/kernel/m
?
+Adam/dense_111/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_111/kernel/m*
_output_shapes
:	'?*
dtype0
?
Adam/dense_111/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_111/bias/m
|
)Adam/dense_111/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_111/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_112/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameAdam/dense_112/kernel/m
?
+Adam/dense_112/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_112/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_111/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	'?*(
shared_nameAdam/dense_111/kernel/v
?
+Adam/dense_111/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_111/kernel/v*
_output_shapes
:	'?*
dtype0
?
Adam/dense_111/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_111/bias/v
|
)Adam/dense_111/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_111/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_112/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameAdam/dense_112/kernel/v
?
+Adam/dense_112/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_112/kernel/v*
_output_shapes
:	?*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

	kernel

bias
	variables
regularization_losses
trainable_variables
	keras_api
^

kernel
	variables
regularization_losses
trainable_variables
	keras_api
v
iter

beta_1

beta_2
	decay
learning_rate	m-
m.m/	v0
v1v2

	0

1
2
 

	0

1
2
?

layers
non_trainable_variables
metrics
	variables
layer_metrics
regularization_losses
layer_regularization_losses
trainable_variables
 
\Z
VARIABLE_VALUEdense_111/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_111/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

	0

1
 

	0

1
?

layers
non_trainable_variables
 metrics
	variables
!layer_metrics
regularization_losses
"layer_regularization_losses
trainable_variables
\Z
VARIABLE_VALUEdense_112/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
?

#layers
$non_trainable_variables
%metrics
	variables
&layer_metrics
regularization_losses
'layer_regularization_losses
trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

(0
 
 
 
 
 
 
 
 
 
 
 
 
4
	)total
	*count
+	variables
,	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

+	variables
}
VARIABLE_VALUEAdam/dense_111/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_111/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_112/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_111/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_111/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_112/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_dense_111_inputPlaceholder*'
_output_shapes
:?????????'*
dtype0*
shape:?????????'
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_111_inputdense_111/kerneldense_111/biasdense_112/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_2068889
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_111/kernel/Read/ReadVariableOp"dense_111/bias/Read/ReadVariableOp$dense_112/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_111/kernel/m/Read/ReadVariableOp)Adam/dense_111/bias/m/Read/ReadVariableOp+Adam/dense_112/kernel/m/Read/ReadVariableOp+Adam/dense_111/kernel/v/Read/ReadVariableOp)Adam/dense_111/bias/v/Read/ReadVariableOp+Adam/dense_112/kernel/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_2069044
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_111/kerneldense_111/biasdense_112/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_111/kernel/mAdam/dense_111/bias/mAdam/dense_112/kernel/mAdam/dense_111/kernel/vAdam/dense_111/bias/vAdam/dense_112/kernel/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_2069102??
?
?
%__inference_signature_wrapper_2068889
dense_111_input
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_111_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_20687582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????':::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:?????????'
)
_user_specified_namedense_111_input
?F
?
#__inference__traced_restore_2069102
file_prefix%
!assignvariableop_dense_111_kernel%
!assignvariableop_1_dense_111_bias'
#assignvariableop_2_dense_112_kernel 
assignvariableop_3_adam_iter"
assignvariableop_4_adam_beta_1"
assignvariableop_5_adam_beta_2!
assignvariableop_6_adam_decay)
%assignvariableop_7_adam_learning_rate
assignvariableop_8_total
assignvariableop_9_count/
+assignvariableop_10_adam_dense_111_kernel_m-
)assignvariableop_11_adam_dense_111_bias_m/
+assignvariableop_12_adam_dense_112_kernel_m/
+assignvariableop_13_adam_dense_111_kernel_v-
)assignvariableop_14_adam_dense_111_bias_v/
+assignvariableop_15_adam_dense_112_kernel_v
identity_17??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_dense_111_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_111_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_112_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp%assignvariableop_7_adam_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp+assignvariableop_10_adam_dense_111_kernel_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp)assignvariableop_11_adam_dense_111_bias_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp+assignvariableop_12_adam_dense_112_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp+assignvariableop_13_adam_dense_111_kernel_vIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_dense_111_bias_vIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_dense_112_kernel_vIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_16?
Identity_17IdentityIdentity_16:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_17"#
identity_17Identity_17:output:0*U
_input_shapesD
B: ::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
J__inference_sequential_60_layer_call_and_return_conditional_losses_2068809
dense_111_input
dense_111_2068784
dense_111_2068786
dense_112_2068805
identity??!dense_111/StatefulPartitionedCall?!dense_112/StatefulPartitionedCall?
!dense_111/StatefulPartitionedCallStatefulPartitionedCalldense_111_inputdense_111_2068784dense_111_2068786*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_111_layer_call_and_return_conditional_losses_20687732#
!dense_111/StatefulPartitionedCall?
!dense_112/StatefulPartitionedCallStatefulPartitionedCall*dense_111/StatefulPartitionedCall:output:0dense_112_2068805*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_112_layer_call_and_return_conditional_losses_20687962#
!dense_112/StatefulPartitionedCall?
IdentityIdentity*dense_112/StatefulPartitionedCall:output:0"^dense_111/StatefulPartitionedCall"^dense_112/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????':::2F
!dense_111/StatefulPartitionedCall!dense_111/StatefulPartitionedCall2F
!dense_112/StatefulPartitionedCall!dense_112/StatefulPartitionedCall:X T
'
_output_shapes
:?????????'
)
_user_specified_namedense_111_input
?
?
/__inference_sequential_60_layer_call_fn_2068868
dense_111_input
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_111_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_60_layer_call_and_return_conditional_losses_20688592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????':::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:?????????'
)
_user_specified_namedense_111_input
?
q
+__inference_dense_112_layer_call_fn_2068973

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_112_layer_call_and_return_conditional_losses_20687962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_60_layer_call_and_return_conditional_losses_2068836

inputs
dense_111_2068827
dense_111_2068829
dense_112_2068832
identity??!dense_111/StatefulPartitionedCall?!dense_112/StatefulPartitionedCall?
!dense_111/StatefulPartitionedCallStatefulPartitionedCallinputsdense_111_2068827dense_111_2068829*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_111_layer_call_and_return_conditional_losses_20687732#
!dense_111/StatefulPartitionedCall?
!dense_112/StatefulPartitionedCallStatefulPartitionedCall*dense_111/StatefulPartitionedCall:output:0dense_112_2068832*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_112_layer_call_and_return_conditional_losses_20687962#
!dense_112/StatefulPartitionedCall?
IdentityIdentity*dense_112/StatefulPartitionedCall:output:0"^dense_111/StatefulPartitionedCall"^dense_112/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????':::2F
!dense_111/StatefulPartitionedCall!dense_111/StatefulPartitionedCall2F
!dense_112/StatefulPartitionedCall!dense_112/StatefulPartitionedCall:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?
?
F__inference_dense_112_layer_call_and_return_conditional_losses_2068796

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_60_layer_call_and_return_conditional_losses_2068917

inputs,
(dense_111_matmul_readvariableop_resource-
)dense_111_biasadd_readvariableop_resource,
(dense_112_matmul_readvariableop_resource
identity?? dense_111/BiasAdd/ReadVariableOp?dense_111/MatMul/ReadVariableOp?dense_112/MatMul/ReadVariableOp?
dense_111/MatMul/ReadVariableOpReadVariableOp(dense_111_matmul_readvariableop_resource*
_output_shapes
:	'?*
dtype02!
dense_111/MatMul/ReadVariableOp?
dense_111/MatMulMatMulinputs'dense_111/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_111/MatMul?
 dense_111/BiasAdd/ReadVariableOpReadVariableOp)dense_111_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_111/BiasAdd/ReadVariableOp?
dense_111/BiasAddBiasAdddense_111/MatMul:product:0(dense_111/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_111/BiasAddw
dense_111/ReluReludense_111/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_111/Relu?
dense_112/MatMul/ReadVariableOpReadVariableOp(dense_112_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
dense_112/MatMul/ReadVariableOp?
dense_112/MatMulMatMuldense_111/Relu:activations:0'dense_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_112/MatMul?
IdentityIdentitydense_112/MatMul:product:0!^dense_111/BiasAdd/ReadVariableOp ^dense_111/MatMul/ReadVariableOp ^dense_112/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????':::2D
 dense_111/BiasAdd/ReadVariableOp dense_111/BiasAdd/ReadVariableOp2B
dense_111/MatMul/ReadVariableOpdense_111/MatMul/ReadVariableOp2B
dense_112/MatMul/ReadVariableOpdense_112/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?
?
"__inference__wrapped_model_2068758
dense_111_input:
6sequential_60_dense_111_matmul_readvariableop_resource;
7sequential_60_dense_111_biasadd_readvariableop_resource:
6sequential_60_dense_112_matmul_readvariableop_resource
identity??.sequential_60/dense_111/BiasAdd/ReadVariableOp?-sequential_60/dense_111/MatMul/ReadVariableOp?-sequential_60/dense_112/MatMul/ReadVariableOp?
-sequential_60/dense_111/MatMul/ReadVariableOpReadVariableOp6sequential_60_dense_111_matmul_readvariableop_resource*
_output_shapes
:	'?*
dtype02/
-sequential_60/dense_111/MatMul/ReadVariableOp?
sequential_60/dense_111/MatMulMatMuldense_111_input5sequential_60/dense_111/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_60/dense_111/MatMul?
.sequential_60/dense_111/BiasAdd/ReadVariableOpReadVariableOp7sequential_60_dense_111_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.sequential_60/dense_111/BiasAdd/ReadVariableOp?
sequential_60/dense_111/BiasAddBiasAdd(sequential_60/dense_111/MatMul:product:06sequential_60/dense_111/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential_60/dense_111/BiasAdd?
sequential_60/dense_111/ReluRelu(sequential_60/dense_111/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_60/dense_111/Relu?
-sequential_60/dense_112/MatMul/ReadVariableOpReadVariableOp6sequential_60_dense_112_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-sequential_60/dense_112/MatMul/ReadVariableOp?
sequential_60/dense_112/MatMulMatMul*sequential_60/dense_111/Relu:activations:05sequential_60/dense_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_60/dense_112/MatMul?
IdentityIdentity(sequential_60/dense_112/MatMul:product:0/^sequential_60/dense_111/BiasAdd/ReadVariableOp.^sequential_60/dense_111/MatMul/ReadVariableOp.^sequential_60/dense_112/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????':::2`
.sequential_60/dense_111/BiasAdd/ReadVariableOp.sequential_60/dense_111/BiasAdd/ReadVariableOp2^
-sequential_60/dense_111/MatMul/ReadVariableOp-sequential_60/dense_111/MatMul/ReadVariableOp2^
-sequential_60/dense_112/MatMul/ReadVariableOp-sequential_60/dense_112/MatMul/ReadVariableOp:X T
'
_output_shapes
:?????????'
)
_user_specified_namedense_111_input
?
?
/__inference_sequential_60_layer_call_fn_2068939

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_60_layer_call_and_return_conditional_losses_20688592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????':::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?
?
+__inference_dense_111_layer_call_fn_2068959

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_111_layer_call_and_return_conditional_losses_20687732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????'::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?
?
F__inference_dense_112_layer_call_and_return_conditional_losses_2068966

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_111_layer_call_and_return_conditional_losses_2068950

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	'?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????'::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?	
?
F__inference_dense_111_layer_call_and_return_conditional_losses_2068773

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	'?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????'::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?
?
/__inference_sequential_60_layer_call_fn_2068845
dense_111_input
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_111_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_60_layer_call_and_return_conditional_losses_20688362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????':::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:?????????'
)
_user_specified_namedense_111_input
?
?
J__inference_sequential_60_layer_call_and_return_conditional_losses_2068903

inputs,
(dense_111_matmul_readvariableop_resource-
)dense_111_biasadd_readvariableop_resource,
(dense_112_matmul_readvariableop_resource
identity?? dense_111/BiasAdd/ReadVariableOp?dense_111/MatMul/ReadVariableOp?dense_112/MatMul/ReadVariableOp?
dense_111/MatMul/ReadVariableOpReadVariableOp(dense_111_matmul_readvariableop_resource*
_output_shapes
:	'?*
dtype02!
dense_111/MatMul/ReadVariableOp?
dense_111/MatMulMatMulinputs'dense_111/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_111/MatMul?
 dense_111/BiasAdd/ReadVariableOpReadVariableOp)dense_111_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_111/BiasAdd/ReadVariableOp?
dense_111/BiasAddBiasAdddense_111/MatMul:product:0(dense_111/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_111/BiasAddw
dense_111/ReluReludense_111/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_111/Relu?
dense_112/MatMul/ReadVariableOpReadVariableOp(dense_112_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
dense_112/MatMul/ReadVariableOp?
dense_112/MatMulMatMuldense_111/Relu:activations:0'dense_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_112/MatMul?
IdentityIdentitydense_112/MatMul:product:0!^dense_111/BiasAdd/ReadVariableOp ^dense_111/MatMul/ReadVariableOp ^dense_112/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????':::2D
 dense_111/BiasAdd/ReadVariableOp dense_111/BiasAdd/ReadVariableOp2B
dense_111/MatMul/ReadVariableOpdense_111/MatMul/ReadVariableOp2B
dense_112/MatMul/ReadVariableOpdense_112/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?
?
J__inference_sequential_60_layer_call_and_return_conditional_losses_2068859

inputs
dense_111_2068850
dense_111_2068852
dense_112_2068855
identity??!dense_111/StatefulPartitionedCall?!dense_112/StatefulPartitionedCall?
!dense_111/StatefulPartitionedCallStatefulPartitionedCallinputsdense_111_2068850dense_111_2068852*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_111_layer_call_and_return_conditional_losses_20687732#
!dense_111/StatefulPartitionedCall?
!dense_112/StatefulPartitionedCallStatefulPartitionedCall*dense_111/StatefulPartitionedCall:output:0dense_112_2068855*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_112_layer_call_and_return_conditional_losses_20687962#
!dense_112/StatefulPartitionedCall?
IdentityIdentity*dense_112/StatefulPartitionedCall:output:0"^dense_111/StatefulPartitionedCall"^dense_112/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????':::2F
!dense_111/StatefulPartitionedCall!dense_111/StatefulPartitionedCall2F
!dense_112/StatefulPartitionedCall!dense_112/StatefulPartitionedCall:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?+
?
 __inference__traced_save_2069044
file_prefix/
+savev2_dense_111_kernel_read_readvariableop-
)savev2_dense_111_bias_read_readvariableop/
+savev2_dense_112_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_111_kernel_m_read_readvariableop4
0savev2_adam_dense_111_bias_m_read_readvariableop6
2savev2_adam_dense_112_kernel_m_read_readvariableop6
2savev2_adam_dense_111_kernel_v_read_readvariableop4
0savev2_adam_dense_111_bias_v_read_readvariableop6
2savev2_adam_dense_112_kernel_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_111_kernel_read_readvariableop)savev2_dense_111_bias_read_readvariableop+savev2_dense_112_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_111_kernel_m_read_readvariableop0savev2_adam_dense_111_bias_m_read_readvariableop2savev2_adam_dense_112_kernel_m_read_readvariableop2savev2_adam_dense_111_kernel_v_read_readvariableop0savev2_adam_dense_111_bias_v_read_readvariableop2savev2_adam_dense_112_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*|
_input_shapesk
i: :	'?:?:	?: : : : : : : :	'?:?:	?:	'?:?:	?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	'?:!

_output_shapes	
:?:%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :%!

_output_shapes
:	'?:!

_output_shapes	
:?:%!

_output_shapes
:	?:%!

_output_shapes
:	'?:!

_output_shapes	
:?:%!

_output_shapes
:	?:

_output_shapes
: 
?
?
J__inference_sequential_60_layer_call_and_return_conditional_losses_2068821
dense_111_input
dense_111_2068812
dense_111_2068814
dense_112_2068817
identity??!dense_111/StatefulPartitionedCall?!dense_112/StatefulPartitionedCall?
!dense_111/StatefulPartitionedCallStatefulPartitionedCalldense_111_inputdense_111_2068812dense_111_2068814*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_111_layer_call_and_return_conditional_losses_20687732#
!dense_111/StatefulPartitionedCall?
!dense_112/StatefulPartitionedCallStatefulPartitionedCall*dense_111/StatefulPartitionedCall:output:0dense_112_2068817*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_112_layer_call_and_return_conditional_losses_20687962#
!dense_112/StatefulPartitionedCall?
IdentityIdentity*dense_112/StatefulPartitionedCall:output:0"^dense_111/StatefulPartitionedCall"^dense_112/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????':::2F
!dense_111/StatefulPartitionedCall!dense_111/StatefulPartitionedCall2F
!dense_112/StatefulPartitionedCall!dense_112/StatefulPartitionedCall:X T
'
_output_shapes
:?????????'
)
_user_specified_namedense_111_input
?
?
/__inference_sequential_60_layer_call_fn_2068928

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_60_layer_call_and_return_conditional_losses_20688362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????':::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
K
dense_111_input8
!serving_default_dense_111_input:0?????????'=
	dense_1120
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?e
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
3__call__
*4&call_and_return_all_conditional_losses
5_default_save_signature"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_60", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_60", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 39]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_111_input"}}, {"class_name": "Dense", "config": {"name": "dense_111", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 39]}, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_112", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 39}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 39]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_60", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 39]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_111_input"}}, {"class_name": "Dense", "config": {"name": "dense_111", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 39]}, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_112", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "nmse_loss", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

	kernel

bias
	variables
regularization_losses
trainable_variables
	keras_api
6__call__
*7&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_111", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 39]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_111", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 39]}, "dtype": "float32", "units": 200, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 39}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 39]}}
?

kernel
	variables
regularization_losses
trainable_variables
	keras_api
8__call__
*9&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_112", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_112", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
?
iter

beta_1

beta_2
	decay
learning_rate	m-
m.m/	v0
v1v2"
	optimizer
5
	0

1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
	0

1
2"
trackable_list_wrapper
?

layers
non_trainable_variables
metrics
	variables
layer_metrics
regularization_losses
layer_regularization_losses
trainable_variables
3__call__
5_default_save_signature
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
,
:serving_default"
signature_map
#:!	'?2dense_111/kernel
:?2dense_111/bias
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
?

layers
non_trainable_variables
 metrics
	variables
!layer_metrics
regularization_losses
"layer_regularization_losses
trainable_variables
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
#:!	?2dense_112/kernel
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?

#layers
$non_trainable_variables
%metrics
	variables
&layer_metrics
regularization_losses
'layer_regularization_losses
trainable_variables
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
	)total
	*count
+	variables
,	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
)0
*1"
trackable_list_wrapper
-
+	variables"
_generic_user_object
(:&	'?2Adam/dense_111/kernel/m
": ?2Adam/dense_111/bias/m
(:&	?2Adam/dense_112/kernel/m
(:&	'?2Adam/dense_111/kernel/v
": ?2Adam/dense_111/bias/v
(:&	?2Adam/dense_112/kernel/v
?2?
/__inference_sequential_60_layer_call_fn_2068845
/__inference_sequential_60_layer_call_fn_2068928
/__inference_sequential_60_layer_call_fn_2068868
/__inference_sequential_60_layer_call_fn_2068939?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_sequential_60_layer_call_and_return_conditional_losses_2068903
J__inference_sequential_60_layer_call_and_return_conditional_losses_2068809
J__inference_sequential_60_layer_call_and_return_conditional_losses_2068917
J__inference_sequential_60_layer_call_and_return_conditional_losses_2068821?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference__wrapped_model_2068758?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
dense_111_input?????????'
?2?
+__inference_dense_111_layer_call_fn_2068959?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_111_layer_call_and_return_conditional_losses_2068950?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_112_layer_call_fn_2068973?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_112_layer_call_and_return_conditional_losses_2068966?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_2068889dense_111_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_2068758v	
8?5
.?+
)?&
dense_111_input?????????'
? "5?2
0
	dense_112#? 
	dense_112??????????
F__inference_dense_111_layer_call_and_return_conditional_losses_2068950]	
/?,
%?"
 ?
inputs?????????'
? "&?#
?
0??????????
? 
+__inference_dense_111_layer_call_fn_2068959P	
/?,
%?"
 ?
inputs?????????'
? "????????????
F__inference_dense_112_layer_call_and_return_conditional_losses_2068966\0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ~
+__inference_dense_112_layer_call_fn_2068973O0?-
&?#
!?
inputs??????????
? "???????????
J__inference_sequential_60_layer_call_and_return_conditional_losses_2068809n	
@?=
6?3
)?&
dense_111_input?????????'
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_60_layer_call_and_return_conditional_losses_2068821n	
@?=
6?3
)?&
dense_111_input?????????'
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_60_layer_call_and_return_conditional_losses_2068903e	
7?4
-?*
 ?
inputs?????????'
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_60_layer_call_and_return_conditional_losses_2068917e	
7?4
-?*
 ?
inputs?????????'
p 

 
? "%?"
?
0?????????
? ?
/__inference_sequential_60_layer_call_fn_2068845a	
@?=
6?3
)?&
dense_111_input?????????'
p

 
? "???????????
/__inference_sequential_60_layer_call_fn_2068868a	
@?=
6?3
)?&
dense_111_input?????????'
p 

 
? "???????????
/__inference_sequential_60_layer_call_fn_2068928X	
7?4
-?*
 ?
inputs?????????'
p

 
? "???????????
/__inference_sequential_60_layer_call_fn_2068939X	
7?4
-?*
 ?
inputs?????????'
p 

 
? "???????????
%__inference_signature_wrapper_2068889?	
K?H
? 
A?>
<
dense_111_input)?&
dense_111_input?????????'"5?2
0
	dense_112#? 
	dense_112?????????