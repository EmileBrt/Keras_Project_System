©»1
ôÄ
.
Abs
x"T
y"T"
Ttype:

2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	
0
Neg
x"T
y"T"
Ttype:
2
	
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
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
2
Round
x"T
y"T"
Ttype:
2
	
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
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
1
Sign
x"T
y"T"
Ttype:
2
	
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8Â.
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0

Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:@*
dtype0

!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_2/beta/v

5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_2/gamma/v

6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
:@*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:@@*
dtype0

!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/v

5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/v

6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:@*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:@@*
dtype0

Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/batch_normalization/beta/v

3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:@*
dtype0

 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/batch_normalization/gamma/v

4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:@*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:@*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:v@*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:v@*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0

Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:@*
dtype0

!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_2/beta/m

5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_2/gamma/m

6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
:@*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:@@*
dtype0

!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/m

5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/m

6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:@*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:@@*
dtype0

Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/batch_normalization/beta/m

3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:@*
dtype0

 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/batch_normalization/gamma/m

4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:@*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:@*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:v@*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:v@*
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
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
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
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:@*
dtype0
¢
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:@*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@@*
dtype0
¢
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@@*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:@*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:@*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:@*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:v@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:v@*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿv
Î
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense_1/kerneldense_1/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betadense_2/kerneldense_2/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betadense_3/kerneldense_3/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_77031

NoOpNoOp
À
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ú
valueïBë Bã
®
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
ò
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
kernel_quantizer_internal
bias_quantizer_internal

quantizers

 kernel
!bias*
Õ
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(axis
	)gamma
*beta
+moving_mean
,moving_variance*
¥
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3_random_generator* 

4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:	quantizer* 
ò
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
Akernel_quantizer_internal
Bbias_quantizer_internal
C
quantizers

Dkernel
Ebias*
Õ
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance*
¥
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
W_random_generator* 

X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^	quantizer* 
ò
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
ekernel_quantizer_internal
fbias_quantizer_internal
g
quantizers

hkernel
ibias*
Õ
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance*
¥
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
{_random_generator* 
 
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	quantizer* 
ý
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel_quantizer_internal
bias_quantizer_internal

quantizers
kernel
	bias*

 0
!1
)2
*3
+4
,5
D6
E7
M8
N9
O10
P11
h12
i13
q14
r15
s16
t17
18
19*
l
 0
!1
)2
*3
D4
E5
M6
N7
h8
i9
q10
r11
12
13*
* 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
å
	iter
beta_1
beta_2

decay
learning_rate m!m)m*mDmEmMmNmhmimqmrm	m	m  v¡!v¢)v£*v¤Dv¥Ev¦Mv§Nv¨hv©ivªqv«rv¬	v­	v®*

 serving_default* 

 0
!1*

 0
!1*
* 

¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

¦trace_0* 

§trace_0* 
* 
* 

0
1* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
)0
*1
+2
,3*

)0
*1*
* 

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

­trace_0
®trace_1* 

¯trace_0
°trace_1* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 

¶trace_0
·trace_1* 

¸trace_0
¹trace_1* 
* 
* 
* 
* 

ºnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses* 

¿trace_0* 

Àtrace_0* 
* 

D0
E1*

D0
E1*
* 

Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*

Ætrace_0* 

Çtrace_0* 
* 
* 

A0
B1* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
M0
N1
O2
P3*

M0
N1*
* 

Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*

Ítrace_0
Îtrace_1* 

Ïtrace_0
Ðtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses* 

Ötrace_0
×trace_1* 

Øtrace_0
Ùtrace_1* 
* 
* 
* 
* 

Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses* 

ßtrace_0* 

àtrace_0* 
* 

h0
i1*

h0
i1*
* 

ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*

ætrace_0* 

çtrace_0* 
* 
* 

e0
f1* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
q0
r1
s2
t3*

q0
r1*
* 

ènon_trainable_variables
élayers
êmetrics
 ëlayer_regularization_losses
ìlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*

ítrace_0
îtrace_1* 

ïtrace_0
ðtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

ñnon_trainable_variables
òlayers
ómetrics
 ôlayer_regularization_losses
õlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses* 

ötrace_0
÷trace_1* 

øtrace_0
ùtrace_1* 
* 
* 
* 
* 

únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

ÿtrace_0* 

trace_0* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
* 

0
1* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
.
+0
,1
O2
P3
s4
t5*
b
0
1
2
3
4
5
6
7
	8

9
10
11
12*

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 

+0
,1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

O0
P1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

s0
t1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count

_fn_kwargs*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/batch_normalization_2/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/batch_normalization_2/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpConst*F
Tin?
=2;	*
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
GPU 2J 8 *'
f"R 
__inference__traced_save_80852
¢
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense_1/kerneldense_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense_2/kerneldense_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancedense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense/kernel/mAdam/dense/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/dense_1/kernel/mAdam/dense_1/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/dense_2/kernel/mAdam/dense_2/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense/kernel/vAdam/dense/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/dense_1/kernel/vAdam/dense_1/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/dense_2/kernel/vAdam/dense_2/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/dense_3/kernel/vAdam/dense_3/bias/v*E
Tin>
<2:*
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_81033ï,
À

^
B__inference_re_lu_2_layer_call_and_return_conditional_losses_80407

inputs
identityF
SignSigninputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
AbsAbsSign:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
subSubsub/x:output:0Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
addAddV2Sign:y:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
TanhTanhinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
NegNegTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mulMulmul/x:output:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
add_1AddV2Neg:y:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
StopGradientStopGradient	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
add_2AddV2Tanh:y:0StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ü:
ë
H__inference_my_sequential_layer_call_and_return_conditional_losses_76502

inputs
dense_75650:v@
dense_75652:@'
batch_normalization_75655:@'
batch_normalization_75657:@'
batch_normalization_75659:@'
batch_normalization_75661:@
dense_1_75932:@@
dense_1_75934:@)
batch_normalization_1_75937:@)
batch_normalization_1_75939:@)
batch_normalization_1_75941:@)
batch_normalization_1_75943:@
dense_2_76214:@@
dense_2_76216:@)
batch_normalization_2_76219:@)
batch_normalization_2_76221:@)
batch_normalization_2_76223:@)
batch_normalization_2_76225:@
dense_3_76496:@
dense_3_76498:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCallá
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_75650dense_75652*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75649ó
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_75655batch_normalization_75657batch_normalization_75659batch_normalization_75661*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_75178ã
dropout/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_75669Ë
re_lu/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_75687
dense_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0dense_1_75932dense_1_75934*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75931
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_75937batch_normalization_1_75939batch_normalization_1_75941batch_normalization_1_75943*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_75260é
dropout_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_75951Ñ
re_lu_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_75969
dense_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0dense_2_76214dense_2_76216*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_76213
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_76219batch_normalization_2_76221batch_normalization_2_76223batch_normalization_2_76225*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_75342é
dropout_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_76233Ñ
re_lu_2/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_76251
dense_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0dense_3_76496dense_3_76498*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_76495w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿv: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_nameinputs
À

^
B__inference_re_lu_1_layer_call_and_return_conditional_losses_80028

inputs
identityF
SignSigninputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
AbsAbsSign:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
subSubsub/x:output:0Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
addAddV2Sign:y:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
TanhTanhinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
NegNegTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mulMulmul/x:output:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
add_1AddV2Neg:y:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
StopGradientStopGradient	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
add_2AddV2Tanh:y:0StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
è¨
ÿ
B__inference_dense_1_layer_call_and_return_conditional_losses_75931

inputs)
readvariableop_resource:@@'
readvariableop_1_resource:@
identity¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@@*
dtype0]
truedivRealDivReadVariableOp:value:0Cast:y:0*
T0*
_output_shapes

:@@@
AbsAbstruediv:z:0*
T0*
_output_shapes

:@@_
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: m
MaxMaxAbs:y:0Max/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Q
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes

:@P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A\
	truediv_1RealDivmul:z:0truediv_1/y:output:0*
T0*
_output_shapes

:@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3T
addAddV2truediv_1:z:0add/y:output:0*
T0*
_output_shapes

:@<
LogLogadd:z:0*
T0*
_output_shapes

:@P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?\
	truediv_2RealDivLog:y:0truediv_2/y:output:0*
T0*
_output_shapes

:@F
RoundRoundtruediv_2:z:0*
T0*
_output_shapes

:@L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @R
Pow_1PowPow_1/x:output:0	Round:y:0*
T0*
_output_shapes

:@B
Abs_1Abstruediv:z:0*
T0*
_output_shapes

:@@S
	truediv_3RealDiv	Abs_1:y:0	Pow_1:z:0*
T0*
_output_shapes

:@@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
add_1AddV2truediv_3:z:0add_1/y:output:0*
T0*
_output_shapes

:@@B
FloorFloor	add_1:z:0*
T0*
_output_shapes

:@@K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@Q
LessLess	Floor:y:0Less/y:output:0*
T0*
_output_shapes

:@@B
SignSigntruediv:z:0*
T0*
_output_shapes

:@@p
ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
	ones_likeFill(ones_like/Shape/shape_as_tensor:output:0ones_like/Const:output:0*
T0*
_output_shapes

:@@L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A[
mul_1Mulones_like:output:0mul_1/y:output:0*
T0*
_output_shapes

:@@P
truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
	truediv_4RealDiv	mul_1:z:0truediv_4/y:output:0*
T0*
_output_shapes

:@@a
SelectV2SelectV2Less:z:0	Floor:y:0truediv_4:z:0*
T0*
_output_shapes

:@@R
mul_2MulSign:y:0SelectV2:output:0*
T0*
_output_shapes

:@@M
Mul_3Multruediv:z:0	mul_2:z:0*
T0*
_output_shapes

:@@`
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: r
MeanMean	Mul_3:z:0Mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(K
Mul_4Mul	mul_2:z:0	mul_2:z:0*
T0*
_output_shapes

:@@b
Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_1Mean	Mul_4:z:0!Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_2AddV2Mean_1:output:0add_2/y:output:0*
T0*
_output_shapes

:@W
	truediv_5RealDivMean:output:0	add_2:z:0*
T0*
_output_shapes

:@L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3X
add_3AddV2truediv_5:z:0add_3/y:output:0*
T0*
_output_shapes

:@@
Log_1Log	add_3:z:0*
T0*
_output_shapes

:@P
truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?^
	truediv_6RealDiv	Log_1:y:0truediv_6/y:output:0*
T0*
_output_shapes

:@H
Round_1Roundtruediv_6:z:0*
T0*
_output_shapes

:@L
Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_2PowPow_2/x:output:0Round_1:y:0*
T0*
_output_shapes

:@B
Abs_2Abstruediv:z:0*
T0*
_output_shapes

:@@S
	truediv_7RealDiv	Abs_2:y:0	Pow_2:z:0*
T0*
_output_shapes

:@@L
add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
add_4AddV2truediv_7:z:0add_4/y:output:0*
T0*
_output_shapes

:@@D
Floor_1Floor	add_4:z:0*
T0*
_output_shapes

:@@M
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_1LessFloor_1:y:0Less_1/y:output:0*
T0*
_output_shapes

:@@D
Sign_1Signtruediv:z:0*
T0*
_output_shapes

:@@r
!ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_1Fill*ones_like_1/Shape/shape_as_tensor:output:0ones_like_1/Const:output:0*
T0*
_output_shapes

:@@L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A]
mul_5Mulones_like_1:output:0mul_5/y:output:0*
T0*
_output_shapes

:@@P
truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
	truediv_8RealDiv	mul_5:z:0truediv_8/y:output:0*
T0*
_output_shapes

:@@g

SelectV2_1SelectV2
Less_1:z:0Floor_1:y:0truediv_8:z:0*
T0*
_output_shapes

:@@V
mul_6Mul
Sign_1:y:0SelectV2_1:output:0*
T0*
_output_shapes

:@@M
Mul_7Multruediv:z:0	mul_6:z:0*
T0*
_output_shapes

:@@b
Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_2Mean	Mul_7:z:0!Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(K
Mul_8Mul	mul_6:z:0	mul_6:z:0*
T0*
_output_shapes

:@@b
Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_3Mean	Mul_8:z:0!Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_5AddV2Mean_3:output:0add_5/y:output:0*
T0*
_output_shapes

:@Y
	truediv_9RealDivMean_2:output:0	add_5:z:0*
T0*
_output_shapes

:@L
add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3X
add_6AddV2truediv_9:z:0add_6/y:output:0*
T0*
_output_shapes

:@@
Log_2Log	add_6:z:0*
T0*
_output_shapes

:@Q
truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_10RealDiv	Log_2:y:0truediv_10/y:output:0*
T0*
_output_shapes

:@I
Round_2Roundtruediv_10:z:0*
T0*
_output_shapes

:@L
Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_3PowPow_3/x:output:0Round_2:y:0*
T0*
_output_shapes

:@B
Abs_3Abstruediv:z:0*
T0*
_output_shapes

:@@T

truediv_11RealDiv	Abs_3:y:0	Pow_3:z:0*
T0*
_output_shapes

:@@L
add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Y
add_7AddV2truediv_11:z:0add_7/y:output:0*
T0*
_output_shapes

:@@D
Floor_2Floor	add_7:z:0*
T0*
_output_shapes

:@@M
Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_2LessFloor_2:y:0Less_2/y:output:0*
T0*
_output_shapes

:@@D
Sign_2Signtruediv:z:0*
T0*
_output_shapes

:@@r
!ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_2Fill*ones_like_2/Shape/shape_as_tensor:output:0ones_like_2/Const:output:0*
T0*
_output_shapes

:@@L
mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A]
mul_9Mulones_like_2:output:0mul_9/y:output:0*
T0*
_output_shapes

:@@Q
truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @`

truediv_12RealDiv	mul_9:z:0truediv_12/y:output:0*
T0*
_output_shapes

:@@h

SelectV2_2SelectV2
Less_2:z:0Floor_2:y:0truediv_12:z:0*
T0*
_output_shapes

:@@W
mul_10Mul
Sign_2:y:0SelectV2_2:output:0*
T0*
_output_shapes

:@@O
Mul_11Multruediv:z:0
mul_10:z:0*
T0*
_output_shapes

:@@b
Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_4Mean
Mul_11:z:0!Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_12Mul
mul_10:z:0
mul_10:z:0*
T0*
_output_shapes

:@@b
Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_5Mean
Mul_12:z:0!Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_8AddV2Mean_5:output:0add_8/y:output:0*
T0*
_output_shapes

:@Z

truediv_13RealDivMean_4:output:0	add_8:z:0*
T0*
_output_shapes

:@L
add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Y
add_9AddV2truediv_13:z:0add_9/y:output:0*
T0*
_output_shapes

:@@
Log_3Log	add_9:z:0*
T0*
_output_shapes

:@Q
truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_14RealDiv	Log_3:y:0truediv_14/y:output:0*
T0*
_output_shapes

:@I
Round_3Roundtruediv_14:z:0*
T0*
_output_shapes

:@L
Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_4PowPow_4/x:output:0Round_3:y:0*
T0*
_output_shapes

:@B
Abs_4Abstruediv:z:0*
T0*
_output_shapes

:@@T

truediv_15RealDiv	Abs_4:y:0	Pow_4:z:0*
T0*
_output_shapes

:@@M
add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_10AddV2truediv_15:z:0add_10/y:output:0*
T0*
_output_shapes

:@@E
Floor_3Floor
add_10:z:0*
T0*
_output_shapes

:@@M
Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_3LessFloor_3:y:0Less_3/y:output:0*
T0*
_output_shapes

:@@D
Sign_3Signtruediv:z:0*
T0*
_output_shapes

:@@r
!ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   V
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_3Fill*ones_like_3/Shape/shape_as_tensor:output:0ones_like_3/Const:output:0*
T0*
_output_shapes

:@@M
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A_
mul_13Mulones_like_3:output:0mul_13/y:output:0*
T0*
_output_shapes

:@@Q
truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @a

truediv_16RealDiv
mul_13:z:0truediv_16/y:output:0*
T0*
_output_shapes

:@@h

SelectV2_3SelectV2
Less_3:z:0Floor_3:y:0truediv_16:z:0*
T0*
_output_shapes

:@@W
mul_14Mul
Sign_3:y:0SelectV2_3:output:0*
T0*
_output_shapes

:@@O
Mul_15Multruediv:z:0
mul_14:z:0*
T0*
_output_shapes

:@@b
Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_6Mean
Mul_15:z:0!Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_16Mul
mul_14:z:0
mul_14:z:0*
T0*
_output_shapes

:@@b
Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_7Mean
Mul_16:z:0!Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(M
add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3\
add_11AddV2Mean_7:output:0add_11/y:output:0*
T0*
_output_shapes

:@[

truediv_17RealDivMean_6:output:0
add_11:z:0*
T0*
_output_shapes

:@M
add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3[
add_12AddV2truediv_17:z:0add_12/y:output:0*
T0*
_output_shapes

:@A
Log_4Log
add_12:z:0*
T0*
_output_shapes

:@Q
truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_18RealDiv	Log_4:y:0truediv_18/y:output:0*
T0*
_output_shapes

:@I
Round_4Roundtruediv_18:z:0*
T0*
_output_shapes

:@L
Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_5PowPow_5/x:output:0Round_4:y:0*
T0*
_output_shapes

:@B
Abs_5Abstruediv:z:0*
T0*
_output_shapes

:@@T

truediv_19RealDiv	Abs_5:y:0	Pow_5:z:0*
T0*
_output_shapes

:@@M
add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_13AddV2truediv_19:z:0add_13/y:output:0*
T0*
_output_shapes

:@@E
Floor_4Floor
add_13:z:0*
T0*
_output_shapes

:@@M
Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_4LessFloor_4:y:0Less_4/y:output:0*
T0*
_output_shapes

:@@D
Sign_4Signtruediv:z:0*
T0*
_output_shapes

:@@r
!ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   V
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_4Fill*ones_like_4/Shape/shape_as_tensor:output:0ones_like_4/Const:output:0*
T0*
_output_shapes

:@@M
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A_
mul_17Mulones_like_4:output:0mul_17/y:output:0*
T0*
_output_shapes

:@@Q
truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @a

truediv_20RealDiv
mul_17:z:0truediv_20/y:output:0*
T0*
_output_shapes

:@@h

SelectV2_4SelectV2
Less_4:z:0Floor_4:y:0truediv_20:z:0*
T0*
_output_shapes

:@@W
mul_18Mul
Sign_4:y:0SelectV2_4:output:0*
T0*
_output_shapes

:@@O
Mul_19Multruediv:z:0
mul_18:z:0*
T0*
_output_shapes

:@@b
Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_8Mean
Mul_19:z:0!Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_20Mul
mul_18:z:0
mul_18:z:0*
T0*
_output_shapes

:@@b
Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_9Mean
Mul_20:z:0!Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(M
add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3\
add_14AddV2Mean_9:output:0add_14/y:output:0*
T0*
_output_shapes

:@[

truediv_21RealDivMean_8:output:0
add_14:z:0*
T0*
_output_shapes

:@M
add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3[
add_15AddV2truediv_21:z:0add_15/y:output:0*
T0*
_output_shapes

:@A
Log_5Log
add_15:z:0*
T0*
_output_shapes

:@Q
truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_22RealDiv	Log_5:y:0truediv_22/y:output:0*
T0*
_output_shapes

:@I
Round_5Roundtruediv_22:z:0*
T0*
_output_shapes

:@L
Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_6PowPow_6/x:output:0Round_5:y:0*
T0*
_output_shapes

:@M
mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AT
mul_21Mul	Pow_6:z:0mul_21/y:output:0*
T0*
_output_shapes

:@M
mul_22MulCast:y:0truediv:z:0*
T0*
_output_shapes

:@@L
mul_23MulCast:y:0
mul_18:z:0*
T0*
_output_shapes

:@@Q
truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aa

truediv_23RealDiv
mul_23:z:0truediv_23/y:output:0*
T0*
_output_shapes

:@@R
mul_24Mul
mul_21:z:0truediv_23:z:0*
T0*
_output_shapes

:@@?
NegNeg
mul_22:z:0*
T0*
_output_shapes

:@@M
add_16AddV2Neg:y:0
mul_24:z:0*
T0*
_output_shapes

:@@M
mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mul_25Mulmul_25/x:output:0
add_16:z:0*
T0*
_output_shapes

:@@Q
StopGradientStopGradient
mul_25:z:0*
T0*
_output_shapes

:@@[
add_17AddV2
mul_22:z:0StopGradient:output:0*
T0*
_output_shapes

:@@V
MatMulMatMulinputs
add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@I
Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_7PowPow_7/x:output:0Pow_7/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0M
mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A_
mul_26MulReadVariableOp_1:value:0mul_26/y:output:0*
T0*
_output_shapes
:@R

truediv_24RealDiv
mul_26:z:0
Cast_1:y:0*
T0*
_output_shapes
:@A
Neg_1Negtruediv_24:z:0*
T0*
_output_shapes
:@E
Round_6Roundtruediv_24:z:0*
T0*
_output_shapes
:@L
add_18AddV2	Neg_1:y:0Round_6:y:0*
T0*
_output_shapes
:@O
StopGradient_1StopGradient
add_18:z:0*
T0*
_output_shapes
:@]
add_19AddV2truediv_24:z:0StopGradient_1:output:0*
T0*
_output_shapes
:@\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@s
clip_by_value/MinimumMinimum
add_19:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:@T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ár
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:@Q
mul_27Mul
Cast_1:y:0clip_by_value:z:0*
T0*
_output_shapes
:@Q
truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A]

truediv_25RealDiv
mul_27:z:0truediv_25/y:output:0*
T0*
_output_shapes
:@M
mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mul_28Mulmul_28/x:output:0truediv_25:z:0*
T0*
_output_shapes
:@f
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0K
Neg_2NegReadVariableOp_2:value:0*
T0*
_output_shapes
:@K
add_20AddV2	Neg_2:y:0
mul_28:z:0*
T0*
_output_shapes
:@M
mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Q
mul_29Mulmul_29/x:output:0
add_20:z:0*
T0*
_output_shapes
:@O
StopGradient_2StopGradient
mul_29:z:0*
T0*
_output_shapes
:@f
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0g
add_21AddV2ReadVariableOp_3:value:0StopGradient_2:output:0*
T0*
_output_shapes
:@b
BiasAddBiasAddMatMul:product:0
add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_3:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ò	
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_76620

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

C
'__inference_dropout_layer_call_fn_79606

inputs
identity­
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_75669`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
%
ç
N__inference_batch_normalization_layer_call_and_return_conditional_losses_75225

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¦
Ð
5__inference_batch_normalization_1_layer_call_fn_79913

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_75260o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

è
#__inference_signature_wrapper_77031
input_1
unknown:v@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_75154o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿv: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!
_user_specified_name	input_1
è¨
ÿ
B__inference_dense_2_layer_call_and_return_conditional_losses_80279

inputs)
readvariableop_resource:@@'
readvariableop_1_resource:@
identity¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@@*
dtype0]
truedivRealDivReadVariableOp:value:0Cast:y:0*
T0*
_output_shapes

:@@@
AbsAbstruediv:z:0*
T0*
_output_shapes

:@@_
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: m
MaxMaxAbs:y:0Max/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Q
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes

:@P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A\
	truediv_1RealDivmul:z:0truediv_1/y:output:0*
T0*
_output_shapes

:@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3T
addAddV2truediv_1:z:0add/y:output:0*
T0*
_output_shapes

:@<
LogLogadd:z:0*
T0*
_output_shapes

:@P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?\
	truediv_2RealDivLog:y:0truediv_2/y:output:0*
T0*
_output_shapes

:@F
RoundRoundtruediv_2:z:0*
T0*
_output_shapes

:@L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @R
Pow_1PowPow_1/x:output:0	Round:y:0*
T0*
_output_shapes

:@B
Abs_1Abstruediv:z:0*
T0*
_output_shapes

:@@S
	truediv_3RealDiv	Abs_1:y:0	Pow_1:z:0*
T0*
_output_shapes

:@@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
add_1AddV2truediv_3:z:0add_1/y:output:0*
T0*
_output_shapes

:@@B
FloorFloor	add_1:z:0*
T0*
_output_shapes

:@@K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@Q
LessLess	Floor:y:0Less/y:output:0*
T0*
_output_shapes

:@@B
SignSigntruediv:z:0*
T0*
_output_shapes

:@@p
ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
	ones_likeFill(ones_like/Shape/shape_as_tensor:output:0ones_like/Const:output:0*
T0*
_output_shapes

:@@L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A[
mul_1Mulones_like:output:0mul_1/y:output:0*
T0*
_output_shapes

:@@P
truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
	truediv_4RealDiv	mul_1:z:0truediv_4/y:output:0*
T0*
_output_shapes

:@@a
SelectV2SelectV2Less:z:0	Floor:y:0truediv_4:z:0*
T0*
_output_shapes

:@@R
mul_2MulSign:y:0SelectV2:output:0*
T0*
_output_shapes

:@@M
Mul_3Multruediv:z:0	mul_2:z:0*
T0*
_output_shapes

:@@`
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: r
MeanMean	Mul_3:z:0Mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(K
Mul_4Mul	mul_2:z:0	mul_2:z:0*
T0*
_output_shapes

:@@b
Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_1Mean	Mul_4:z:0!Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_2AddV2Mean_1:output:0add_2/y:output:0*
T0*
_output_shapes

:@W
	truediv_5RealDivMean:output:0	add_2:z:0*
T0*
_output_shapes

:@L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3X
add_3AddV2truediv_5:z:0add_3/y:output:0*
T0*
_output_shapes

:@@
Log_1Log	add_3:z:0*
T0*
_output_shapes

:@P
truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?^
	truediv_6RealDiv	Log_1:y:0truediv_6/y:output:0*
T0*
_output_shapes

:@H
Round_1Roundtruediv_6:z:0*
T0*
_output_shapes

:@L
Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_2PowPow_2/x:output:0Round_1:y:0*
T0*
_output_shapes

:@B
Abs_2Abstruediv:z:0*
T0*
_output_shapes

:@@S
	truediv_7RealDiv	Abs_2:y:0	Pow_2:z:0*
T0*
_output_shapes

:@@L
add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
add_4AddV2truediv_7:z:0add_4/y:output:0*
T0*
_output_shapes

:@@D
Floor_1Floor	add_4:z:0*
T0*
_output_shapes

:@@M
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_1LessFloor_1:y:0Less_1/y:output:0*
T0*
_output_shapes

:@@D
Sign_1Signtruediv:z:0*
T0*
_output_shapes

:@@r
!ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_1Fill*ones_like_1/Shape/shape_as_tensor:output:0ones_like_1/Const:output:0*
T0*
_output_shapes

:@@L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A]
mul_5Mulones_like_1:output:0mul_5/y:output:0*
T0*
_output_shapes

:@@P
truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
	truediv_8RealDiv	mul_5:z:0truediv_8/y:output:0*
T0*
_output_shapes

:@@g

SelectV2_1SelectV2
Less_1:z:0Floor_1:y:0truediv_8:z:0*
T0*
_output_shapes

:@@V
mul_6Mul
Sign_1:y:0SelectV2_1:output:0*
T0*
_output_shapes

:@@M
Mul_7Multruediv:z:0	mul_6:z:0*
T0*
_output_shapes

:@@b
Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_2Mean	Mul_7:z:0!Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(K
Mul_8Mul	mul_6:z:0	mul_6:z:0*
T0*
_output_shapes

:@@b
Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_3Mean	Mul_8:z:0!Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_5AddV2Mean_3:output:0add_5/y:output:0*
T0*
_output_shapes

:@Y
	truediv_9RealDivMean_2:output:0	add_5:z:0*
T0*
_output_shapes

:@L
add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3X
add_6AddV2truediv_9:z:0add_6/y:output:0*
T0*
_output_shapes

:@@
Log_2Log	add_6:z:0*
T0*
_output_shapes

:@Q
truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_10RealDiv	Log_2:y:0truediv_10/y:output:0*
T0*
_output_shapes

:@I
Round_2Roundtruediv_10:z:0*
T0*
_output_shapes

:@L
Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_3PowPow_3/x:output:0Round_2:y:0*
T0*
_output_shapes

:@B
Abs_3Abstruediv:z:0*
T0*
_output_shapes

:@@T

truediv_11RealDiv	Abs_3:y:0	Pow_3:z:0*
T0*
_output_shapes

:@@L
add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Y
add_7AddV2truediv_11:z:0add_7/y:output:0*
T0*
_output_shapes

:@@D
Floor_2Floor	add_7:z:0*
T0*
_output_shapes

:@@M
Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_2LessFloor_2:y:0Less_2/y:output:0*
T0*
_output_shapes

:@@D
Sign_2Signtruediv:z:0*
T0*
_output_shapes

:@@r
!ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_2Fill*ones_like_2/Shape/shape_as_tensor:output:0ones_like_2/Const:output:0*
T0*
_output_shapes

:@@L
mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A]
mul_9Mulones_like_2:output:0mul_9/y:output:0*
T0*
_output_shapes

:@@Q
truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @`

truediv_12RealDiv	mul_9:z:0truediv_12/y:output:0*
T0*
_output_shapes

:@@h

SelectV2_2SelectV2
Less_2:z:0Floor_2:y:0truediv_12:z:0*
T0*
_output_shapes

:@@W
mul_10Mul
Sign_2:y:0SelectV2_2:output:0*
T0*
_output_shapes

:@@O
Mul_11Multruediv:z:0
mul_10:z:0*
T0*
_output_shapes

:@@b
Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_4Mean
Mul_11:z:0!Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_12Mul
mul_10:z:0
mul_10:z:0*
T0*
_output_shapes

:@@b
Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_5Mean
Mul_12:z:0!Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_8AddV2Mean_5:output:0add_8/y:output:0*
T0*
_output_shapes

:@Z

truediv_13RealDivMean_4:output:0	add_8:z:0*
T0*
_output_shapes

:@L
add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Y
add_9AddV2truediv_13:z:0add_9/y:output:0*
T0*
_output_shapes

:@@
Log_3Log	add_9:z:0*
T0*
_output_shapes

:@Q
truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_14RealDiv	Log_3:y:0truediv_14/y:output:0*
T0*
_output_shapes

:@I
Round_3Roundtruediv_14:z:0*
T0*
_output_shapes

:@L
Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_4PowPow_4/x:output:0Round_3:y:0*
T0*
_output_shapes

:@B
Abs_4Abstruediv:z:0*
T0*
_output_shapes

:@@T

truediv_15RealDiv	Abs_4:y:0	Pow_4:z:0*
T0*
_output_shapes

:@@M
add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_10AddV2truediv_15:z:0add_10/y:output:0*
T0*
_output_shapes

:@@E
Floor_3Floor
add_10:z:0*
T0*
_output_shapes

:@@M
Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_3LessFloor_3:y:0Less_3/y:output:0*
T0*
_output_shapes

:@@D
Sign_3Signtruediv:z:0*
T0*
_output_shapes

:@@r
!ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   V
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_3Fill*ones_like_3/Shape/shape_as_tensor:output:0ones_like_3/Const:output:0*
T0*
_output_shapes

:@@M
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A_
mul_13Mulones_like_3:output:0mul_13/y:output:0*
T0*
_output_shapes

:@@Q
truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @a

truediv_16RealDiv
mul_13:z:0truediv_16/y:output:0*
T0*
_output_shapes

:@@h

SelectV2_3SelectV2
Less_3:z:0Floor_3:y:0truediv_16:z:0*
T0*
_output_shapes

:@@W
mul_14Mul
Sign_3:y:0SelectV2_3:output:0*
T0*
_output_shapes

:@@O
Mul_15Multruediv:z:0
mul_14:z:0*
T0*
_output_shapes

:@@b
Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_6Mean
Mul_15:z:0!Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_16Mul
mul_14:z:0
mul_14:z:0*
T0*
_output_shapes

:@@b
Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_7Mean
Mul_16:z:0!Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(M
add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3\
add_11AddV2Mean_7:output:0add_11/y:output:0*
T0*
_output_shapes

:@[

truediv_17RealDivMean_6:output:0
add_11:z:0*
T0*
_output_shapes

:@M
add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3[
add_12AddV2truediv_17:z:0add_12/y:output:0*
T0*
_output_shapes

:@A
Log_4Log
add_12:z:0*
T0*
_output_shapes

:@Q
truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_18RealDiv	Log_4:y:0truediv_18/y:output:0*
T0*
_output_shapes

:@I
Round_4Roundtruediv_18:z:0*
T0*
_output_shapes

:@L
Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_5PowPow_5/x:output:0Round_4:y:0*
T0*
_output_shapes

:@B
Abs_5Abstruediv:z:0*
T0*
_output_shapes

:@@T

truediv_19RealDiv	Abs_5:y:0	Pow_5:z:0*
T0*
_output_shapes

:@@M
add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_13AddV2truediv_19:z:0add_13/y:output:0*
T0*
_output_shapes

:@@E
Floor_4Floor
add_13:z:0*
T0*
_output_shapes

:@@M
Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_4LessFloor_4:y:0Less_4/y:output:0*
T0*
_output_shapes

:@@D
Sign_4Signtruediv:z:0*
T0*
_output_shapes

:@@r
!ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   V
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_4Fill*ones_like_4/Shape/shape_as_tensor:output:0ones_like_4/Const:output:0*
T0*
_output_shapes

:@@M
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A_
mul_17Mulones_like_4:output:0mul_17/y:output:0*
T0*
_output_shapes

:@@Q
truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @a

truediv_20RealDiv
mul_17:z:0truediv_20/y:output:0*
T0*
_output_shapes

:@@h

SelectV2_4SelectV2
Less_4:z:0Floor_4:y:0truediv_20:z:0*
T0*
_output_shapes

:@@W
mul_18Mul
Sign_4:y:0SelectV2_4:output:0*
T0*
_output_shapes

:@@O
Mul_19Multruediv:z:0
mul_18:z:0*
T0*
_output_shapes

:@@b
Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_8Mean
Mul_19:z:0!Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_20Mul
mul_18:z:0
mul_18:z:0*
T0*
_output_shapes

:@@b
Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_9Mean
Mul_20:z:0!Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(M
add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3\
add_14AddV2Mean_9:output:0add_14/y:output:0*
T0*
_output_shapes

:@[

truediv_21RealDivMean_8:output:0
add_14:z:0*
T0*
_output_shapes

:@M
add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3[
add_15AddV2truediv_21:z:0add_15/y:output:0*
T0*
_output_shapes

:@A
Log_5Log
add_15:z:0*
T0*
_output_shapes

:@Q
truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_22RealDiv	Log_5:y:0truediv_22/y:output:0*
T0*
_output_shapes

:@I
Round_5Roundtruediv_22:z:0*
T0*
_output_shapes

:@L
Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_6PowPow_6/x:output:0Round_5:y:0*
T0*
_output_shapes

:@M
mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AT
mul_21Mul	Pow_6:z:0mul_21/y:output:0*
T0*
_output_shapes

:@M
mul_22MulCast:y:0truediv:z:0*
T0*
_output_shapes

:@@L
mul_23MulCast:y:0
mul_18:z:0*
T0*
_output_shapes

:@@Q
truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aa

truediv_23RealDiv
mul_23:z:0truediv_23/y:output:0*
T0*
_output_shapes

:@@R
mul_24Mul
mul_21:z:0truediv_23:z:0*
T0*
_output_shapes

:@@?
NegNeg
mul_22:z:0*
T0*
_output_shapes

:@@M
add_16AddV2Neg:y:0
mul_24:z:0*
T0*
_output_shapes

:@@M
mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mul_25Mulmul_25/x:output:0
add_16:z:0*
T0*
_output_shapes

:@@Q
StopGradientStopGradient
mul_25:z:0*
T0*
_output_shapes

:@@[
add_17AddV2
mul_22:z:0StopGradient:output:0*
T0*
_output_shapes

:@@V
MatMulMatMulinputs
add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@I
Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_7PowPow_7/x:output:0Pow_7/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0M
mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A_
mul_26MulReadVariableOp_1:value:0mul_26/y:output:0*
T0*
_output_shapes
:@R

truediv_24RealDiv
mul_26:z:0
Cast_1:y:0*
T0*
_output_shapes
:@A
Neg_1Negtruediv_24:z:0*
T0*
_output_shapes
:@E
Round_6Roundtruediv_24:z:0*
T0*
_output_shapes
:@L
add_18AddV2	Neg_1:y:0Round_6:y:0*
T0*
_output_shapes
:@O
StopGradient_1StopGradient
add_18:z:0*
T0*
_output_shapes
:@]
add_19AddV2truediv_24:z:0StopGradient_1:output:0*
T0*
_output_shapes
:@\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@s
clip_by_value/MinimumMinimum
add_19:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:@T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ár
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:@Q
mul_27Mul
Cast_1:y:0clip_by_value:z:0*
T0*
_output_shapes
:@Q
truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A]

truediv_25RealDiv
mul_27:z:0truediv_25/y:output:0*
T0*
_output_shapes
:@M
mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mul_28Mulmul_28/x:output:0truediv_25:z:0*
T0*
_output_shapes
:@f
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0K
Neg_2NegReadVariableOp_2:value:0*
T0*
_output_shapes
:@K
add_20AddV2	Neg_2:y:0
mul_28:z:0*
T0*
_output_shapes
:@M
mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Q
mul_29Mulmul_29/x:output:0
add_20:z:0*
T0*
_output_shapes
:@O
StopGradient_2StopGradient
mul_29:z:0*
T0*
_output_shapes
:@f
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0g
add_21AddV2ReadVariableOp_3:value:0StopGradient_2:output:0*
T0*
_output_shapes
:@b
BiasAddBiasAddMatMul:product:0
add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_3:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
%
é
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_75307

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¢
Î
3__inference_batch_normalization_layer_call_fn_79534

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_75178o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Äß

H__inference_my_sequential_layer_call_and_return_conditional_losses_79270

inputs/
dense_readvariableop_resource:v@-
dense_readvariableop_1_resource:@I
;batch_normalization_assignmovingavg_readvariableop_resource:@K
=batch_normalization_assignmovingavg_1_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@1
dense_1_readvariableop_resource:@@/
!dense_1_readvariableop_1_resource:@K
=batch_normalization_1_assignmovingavg_readvariableop_resource:@M
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@1
dense_2_readvariableop_resource:@@/
!dense_2_readvariableop_1_resource:@K
=batch_normalization_2_assignmovingavg_readvariableop_resource:@M
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_2_batchnorm_readvariableop_resource:@1
dense_3_readvariableop_resource:@/
!dense_3_readvariableop_1_resource:
identity¢#batch_normalization/AssignMovingAvg¢2batch_normalization/AssignMovingAvg/ReadVariableOp¢%batch_normalization/AssignMovingAvg_1¢4batch_normalization/AssignMovingAvg_1/ReadVariableOp¢,batch_normalization/batchnorm/ReadVariableOp¢0batch_normalization/batchnorm/mul/ReadVariableOp¢%batch_normalization_1/AssignMovingAvg¢4batch_normalization_1/AssignMovingAvg/ReadVariableOp¢'batch_normalization_1/AssignMovingAvg_1¢6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_1/batchnorm/ReadVariableOp¢2batch_normalization_1/batchnorm/mul/ReadVariableOp¢%batch_normalization_2/AssignMovingAvg¢4batch_normalization_2/AssignMovingAvg/ReadVariableOp¢'batch_normalization_2/AssignMovingAvg_1¢6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_2/batchnorm/ReadVariableOp¢2batch_normalization_2/batchnorm/mul/ReadVariableOp¢dense/ReadVariableOp¢dense/ReadVariableOp_1¢dense/ReadVariableOp_2¢dense/ReadVariableOp_3¢dense_1/ReadVariableOp¢dense_1/ReadVariableOp_1¢dense_1/ReadVariableOp_2¢dense_1/ReadVariableOp_3¢dense_2/ReadVariableOp¢dense_2/ReadVariableOp_1¢dense_2/ReadVariableOp_2¢dense_2/ReadVariableOp_3¢dense_3/ReadVariableOp¢dense_3/ReadVariableOp_1¢dense_3/ReadVariableOp_2¢dense_3/ReadVariableOp_3M
dense/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :M
dense/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : ]
	dense/PowPowdense/Pow/x:output:0dense/Pow/y:output:0*
T0*
_output_shapes
: Q

dense/CastCastdense/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: r
dense/ReadVariableOpReadVariableOpdense_readvariableop_resource*
_output_shapes

:v@*
dtype0o
dense/truedivRealDivdense/ReadVariableOp:value:0dense/Cast:y:0*
T0*
_output_shapes

:v@L
	dense/AbsAbsdense/truediv:z:0*
T0*
_output_shapes

:v@e
dense/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
	dense/MaxMaxdense/Abs:y:0$dense/Max/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(P
dense/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @c
	dense/mulMuldense/Max:output:0dense/mul/y:output:0*
T0*
_output_shapes

:@V
dense/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `An
dense/truediv_1RealDivdense/mul:z:0dense/truediv_1/y:output:0*
T0*
_output_shapes

:@P
dense/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3f
	dense/addAddV2dense/truediv_1:z:0dense/add/y:output:0*
T0*
_output_shapes

:@H
	dense/LogLogdense/add:z:0*
T0*
_output_shapes

:@V
dense/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?n
dense/truediv_2RealDivdense/Log:y:0dense/truediv_2/y:output:0*
T0*
_output_shapes

:@R
dense/RoundRounddense/truediv_2:z:0*
T0*
_output_shapes

:@R
dense/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dense/Pow_1Powdense/Pow_1/x:output:0dense/Round:y:0*
T0*
_output_shapes

:@N
dense/Abs_1Absdense/truediv:z:0*
T0*
_output_shapes

:v@e
dense/truediv_3RealDivdense/Abs_1:y:0dense/Pow_1:z:0*
T0*
_output_shapes

:v@R
dense/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?j
dense/add_1AddV2dense/truediv_3:z:0dense/add_1/y:output:0*
T0*
_output_shapes

:v@N
dense/FloorFloordense/add_1:z:0*
T0*
_output_shapes

:v@Q
dense/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@c

dense/LessLessdense/Floor:y:0dense/Less/y:output:0*
T0*
_output_shapes

:v@N

dense/SignSigndense/truediv:z:0*
T0*
_output_shapes

:v@v
%dense/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   Z
dense/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense/ones_likeFill.dense/ones_like/Shape/shape_as_tensor:output:0dense/ones_like/Const:output:0*
T0*
_output_shapes

:v@R
dense/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Am
dense/mul_1Muldense/ones_like:output:0dense/mul_1/y:output:0*
T0*
_output_shapes

:v@V
dense/truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @p
dense/truediv_4RealDivdense/mul_1:z:0dense/truediv_4/y:output:0*
T0*
_output_shapes

:v@y
dense/SelectV2SelectV2dense/Less:z:0dense/Floor:y:0dense/truediv_4:z:0*
T0*
_output_shapes

:v@d
dense/mul_2Muldense/Sign:y:0dense/SelectV2:output:0*
T0*
_output_shapes

:v@_
dense/Mul_3Muldense/truediv:z:0dense/mul_2:z:0*
T0*
_output_shapes

:v@f
dense/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 

dense/MeanMeandense/Mul_3:z:0%dense/Mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(]
dense/Mul_4Muldense/mul_2:z:0dense/mul_2:z:0*
T0*
_output_shapes

:v@h
dense/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_1Meandense/Mul_4:z:0'dense/Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(R
dense/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3l
dense/add_2AddV2dense/Mean_1:output:0dense/add_2/y:output:0*
T0*
_output_shapes

:@i
dense/truediv_5RealDivdense/Mean:output:0dense/add_2:z:0*
T0*
_output_shapes

:@R
dense/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3j
dense/add_3AddV2dense/truediv_5:z:0dense/add_3/y:output:0*
T0*
_output_shapes

:@L
dense/Log_1Logdense/add_3:z:0*
T0*
_output_shapes

:@V
dense/truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?p
dense/truediv_6RealDivdense/Log_1:y:0dense/truediv_6/y:output:0*
T0*
_output_shapes

:@T
dense/Round_1Rounddense/truediv_6:z:0*
T0*
_output_shapes

:@R
dense/Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dense/Pow_2Powdense/Pow_2/x:output:0dense/Round_1:y:0*
T0*
_output_shapes

:@N
dense/Abs_2Absdense/truediv:z:0*
T0*
_output_shapes

:v@e
dense/truediv_7RealDivdense/Abs_2:y:0dense/Pow_2:z:0*
T0*
_output_shapes

:v@R
dense/add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?j
dense/add_4AddV2dense/truediv_7:z:0dense/add_4/y:output:0*
T0*
_output_shapes

:v@P
dense/Floor_1Floordense/add_4:z:0*
T0*
_output_shapes

:v@S
dense/Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@i
dense/Less_1Lessdense/Floor_1:y:0dense/Less_1/y:output:0*
T0*
_output_shapes

:v@P
dense/Sign_1Signdense/truediv:z:0*
T0*
_output_shapes

:v@x
'dense/ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   \
dense/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense/ones_like_1Fill0dense/ones_like_1/Shape/shape_as_tensor:output:0 dense/ones_like_1/Const:output:0*
T0*
_output_shapes

:v@R
dense/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Ao
dense/mul_5Muldense/ones_like_1:output:0dense/mul_5/y:output:0*
T0*
_output_shapes

:v@V
dense/truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @p
dense/truediv_8RealDivdense/mul_5:z:0dense/truediv_8/y:output:0*
T0*
_output_shapes

:v@
dense/SelectV2_1SelectV2dense/Less_1:z:0dense/Floor_1:y:0dense/truediv_8:z:0*
T0*
_output_shapes

:v@h
dense/mul_6Muldense/Sign_1:y:0dense/SelectV2_1:output:0*
T0*
_output_shapes

:v@_
dense/Mul_7Muldense/truediv:z:0dense/mul_6:z:0*
T0*
_output_shapes

:v@h
dense/Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_2Meandense/Mul_7:z:0'dense/Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(]
dense/Mul_8Muldense/mul_6:z:0dense/mul_6:z:0*
T0*
_output_shapes

:v@h
dense/Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_3Meandense/Mul_8:z:0'dense/Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(R
dense/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3l
dense/add_5AddV2dense/Mean_3:output:0dense/add_5/y:output:0*
T0*
_output_shapes

:@k
dense/truediv_9RealDivdense/Mean_2:output:0dense/add_5:z:0*
T0*
_output_shapes

:@R
dense/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3j
dense/add_6AddV2dense/truediv_9:z:0dense/add_6/y:output:0*
T0*
_output_shapes

:@L
dense/Log_2Logdense/add_6:z:0*
T0*
_output_shapes

:@W
dense/truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?r
dense/truediv_10RealDivdense/Log_2:y:0dense/truediv_10/y:output:0*
T0*
_output_shapes

:@U
dense/Round_2Rounddense/truediv_10:z:0*
T0*
_output_shapes

:@R
dense/Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dense/Pow_3Powdense/Pow_3/x:output:0dense/Round_2:y:0*
T0*
_output_shapes

:@N
dense/Abs_3Absdense/truediv:z:0*
T0*
_output_shapes

:v@f
dense/truediv_11RealDivdense/Abs_3:y:0dense/Pow_3:z:0*
T0*
_output_shapes

:v@R
dense/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?k
dense/add_7AddV2dense/truediv_11:z:0dense/add_7/y:output:0*
T0*
_output_shapes

:v@P
dense/Floor_2Floordense/add_7:z:0*
T0*
_output_shapes

:v@S
dense/Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@i
dense/Less_2Lessdense/Floor_2:y:0dense/Less_2/y:output:0*
T0*
_output_shapes

:v@P
dense/Sign_2Signdense/truediv:z:0*
T0*
_output_shapes

:v@x
'dense/ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   \
dense/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense/ones_like_2Fill0dense/ones_like_2/Shape/shape_as_tensor:output:0 dense/ones_like_2/Const:output:0*
T0*
_output_shapes

:v@R
dense/mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Ao
dense/mul_9Muldense/ones_like_2:output:0dense/mul_9/y:output:0*
T0*
_output_shapes

:v@W
dense/truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @r
dense/truediv_12RealDivdense/mul_9:z:0dense/truediv_12/y:output:0*
T0*
_output_shapes

:v@
dense/SelectV2_2SelectV2dense/Less_2:z:0dense/Floor_2:y:0dense/truediv_12:z:0*
T0*
_output_shapes

:v@i
dense/mul_10Muldense/Sign_2:y:0dense/SelectV2_2:output:0*
T0*
_output_shapes

:v@a
dense/Mul_11Muldense/truediv:z:0dense/mul_10:z:0*
T0*
_output_shapes

:v@h
dense/Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_4Meandense/Mul_11:z:0'dense/Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(`
dense/Mul_12Muldense/mul_10:z:0dense/mul_10:z:0*
T0*
_output_shapes

:v@h
dense/Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_5Meandense/Mul_12:z:0'dense/Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(R
dense/add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3l
dense/add_8AddV2dense/Mean_5:output:0dense/add_8/y:output:0*
T0*
_output_shapes

:@l
dense/truediv_13RealDivdense/Mean_4:output:0dense/add_8:z:0*
T0*
_output_shapes

:@R
dense/add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3k
dense/add_9AddV2dense/truediv_13:z:0dense/add_9/y:output:0*
T0*
_output_shapes

:@L
dense/Log_3Logdense/add_9:z:0*
T0*
_output_shapes

:@W
dense/truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?r
dense/truediv_14RealDivdense/Log_3:y:0dense/truediv_14/y:output:0*
T0*
_output_shapes

:@U
dense/Round_3Rounddense/truediv_14:z:0*
T0*
_output_shapes

:@R
dense/Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dense/Pow_4Powdense/Pow_4/x:output:0dense/Round_3:y:0*
T0*
_output_shapes

:@N
dense/Abs_4Absdense/truediv:z:0*
T0*
_output_shapes

:v@f
dense/truediv_15RealDivdense/Abs_4:y:0dense/Pow_4:z:0*
T0*
_output_shapes

:v@S
dense/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?m
dense/add_10AddV2dense/truediv_15:z:0dense/add_10/y:output:0*
T0*
_output_shapes

:v@Q
dense/Floor_3Floordense/add_10:z:0*
T0*
_output_shapes

:v@S
dense/Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@i
dense/Less_3Lessdense/Floor_3:y:0dense/Less_3/y:output:0*
T0*
_output_shapes

:v@P
dense/Sign_3Signdense/truediv:z:0*
T0*
_output_shapes

:v@x
'dense/ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   \
dense/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense/ones_like_3Fill0dense/ones_like_3/Shape/shape_as_tensor:output:0 dense/ones_like_3/Const:output:0*
T0*
_output_shapes

:v@S
dense/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Aq
dense/mul_13Muldense/ones_like_3:output:0dense/mul_13/y:output:0*
T0*
_output_shapes

:v@W
dense/truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @s
dense/truediv_16RealDivdense/mul_13:z:0dense/truediv_16/y:output:0*
T0*
_output_shapes

:v@
dense/SelectV2_3SelectV2dense/Less_3:z:0dense/Floor_3:y:0dense/truediv_16:z:0*
T0*
_output_shapes

:v@i
dense/mul_14Muldense/Sign_3:y:0dense/SelectV2_3:output:0*
T0*
_output_shapes

:v@a
dense/Mul_15Muldense/truediv:z:0dense/mul_14:z:0*
T0*
_output_shapes

:v@h
dense/Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_6Meandense/Mul_15:z:0'dense/Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(`
dense/Mul_16Muldense/mul_14:z:0dense/mul_14:z:0*
T0*
_output_shapes

:v@h
dense/Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_7Meandense/Mul_16:z:0'dense/Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(S
dense/add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3n
dense/add_11AddV2dense/Mean_7:output:0dense/add_11/y:output:0*
T0*
_output_shapes

:@m
dense/truediv_17RealDivdense/Mean_6:output:0dense/add_11:z:0*
T0*
_output_shapes

:@S
dense/add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3m
dense/add_12AddV2dense/truediv_17:z:0dense/add_12/y:output:0*
T0*
_output_shapes

:@M
dense/Log_4Logdense/add_12:z:0*
T0*
_output_shapes

:@W
dense/truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?r
dense/truediv_18RealDivdense/Log_4:y:0dense/truediv_18/y:output:0*
T0*
_output_shapes

:@U
dense/Round_4Rounddense/truediv_18:z:0*
T0*
_output_shapes

:@R
dense/Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dense/Pow_5Powdense/Pow_5/x:output:0dense/Round_4:y:0*
T0*
_output_shapes

:@N
dense/Abs_5Absdense/truediv:z:0*
T0*
_output_shapes

:v@f
dense/truediv_19RealDivdense/Abs_5:y:0dense/Pow_5:z:0*
T0*
_output_shapes

:v@S
dense/add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?m
dense/add_13AddV2dense/truediv_19:z:0dense/add_13/y:output:0*
T0*
_output_shapes

:v@Q
dense/Floor_4Floordense/add_13:z:0*
T0*
_output_shapes

:v@S
dense/Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@i
dense/Less_4Lessdense/Floor_4:y:0dense/Less_4/y:output:0*
T0*
_output_shapes

:v@P
dense/Sign_4Signdense/truediv:z:0*
T0*
_output_shapes

:v@x
'dense/ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   \
dense/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense/ones_like_4Fill0dense/ones_like_4/Shape/shape_as_tensor:output:0 dense/ones_like_4/Const:output:0*
T0*
_output_shapes

:v@S
dense/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Aq
dense/mul_17Muldense/ones_like_4:output:0dense/mul_17/y:output:0*
T0*
_output_shapes

:v@W
dense/truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @s
dense/truediv_20RealDivdense/mul_17:z:0dense/truediv_20/y:output:0*
T0*
_output_shapes

:v@
dense/SelectV2_4SelectV2dense/Less_4:z:0dense/Floor_4:y:0dense/truediv_20:z:0*
T0*
_output_shapes

:v@i
dense/mul_18Muldense/Sign_4:y:0dense/SelectV2_4:output:0*
T0*
_output_shapes

:v@a
dense/Mul_19Muldense/truediv:z:0dense/mul_18:z:0*
T0*
_output_shapes

:v@h
dense/Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_8Meandense/Mul_19:z:0'dense/Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(`
dense/Mul_20Muldense/mul_18:z:0dense/mul_18:z:0*
T0*
_output_shapes

:v@h
dense/Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_9Meandense/Mul_20:z:0'dense/Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(S
dense/add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3n
dense/add_14AddV2dense/Mean_9:output:0dense/add_14/y:output:0*
T0*
_output_shapes

:@m
dense/truediv_21RealDivdense/Mean_8:output:0dense/add_14:z:0*
T0*
_output_shapes

:@S
dense/add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3m
dense/add_15AddV2dense/truediv_21:z:0dense/add_15/y:output:0*
T0*
_output_shapes

:@M
dense/Log_5Logdense/add_15:z:0*
T0*
_output_shapes

:@W
dense/truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?r
dense/truediv_22RealDivdense/Log_5:y:0dense/truediv_22/y:output:0*
T0*
_output_shapes

:@U
dense/Round_5Rounddense/truediv_22:z:0*
T0*
_output_shapes

:@R
dense/Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dense/Pow_6Powdense/Pow_6/x:output:0dense/Round_5:y:0*
T0*
_output_shapes

:@S
dense/mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Af
dense/mul_21Muldense/Pow_6:z:0dense/mul_21/y:output:0*
T0*
_output_shapes

:@_
dense/mul_22Muldense/Cast:y:0dense/truediv:z:0*
T0*
_output_shapes

:v@^
dense/mul_23Muldense/Cast:y:0dense/mul_18:z:0*
T0*
_output_shapes

:v@W
dense/truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   As
dense/truediv_23RealDivdense/mul_23:z:0dense/truediv_23/y:output:0*
T0*
_output_shapes

:v@d
dense/mul_24Muldense/mul_21:z:0dense/truediv_23:z:0*
T0*
_output_shapes

:v@K
	dense/NegNegdense/mul_22:z:0*
T0*
_output_shapes

:v@_
dense/add_16AddV2dense/Neg:y:0dense/mul_24:z:0*
T0*
_output_shapes

:v@S
dense/mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
dense/mul_25Muldense/mul_25/x:output:0dense/add_16:z:0*
T0*
_output_shapes

:v@]
dense/StopGradientStopGradientdense/mul_25:z:0*
T0*
_output_shapes

:v@m
dense/add_17AddV2dense/mul_22:z:0dense/StopGradient:output:0*
T0*
_output_shapes

:v@b
dense/MatMulMatMulinputsdense/add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@O
dense/Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :O
dense/Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : c
dense/Pow_7Powdense/Pow_7/x:output:0dense/Pow_7/y:output:0*
T0*
_output_shapes
: U
dense/Cast_1Castdense/Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: r
dense/ReadVariableOp_1ReadVariableOpdense_readvariableop_1_resource*
_output_shapes
:@*
dtype0S
dense/mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aq
dense/mul_26Muldense/ReadVariableOp_1:value:0dense/mul_26/y:output:0*
T0*
_output_shapes
:@d
dense/truediv_24RealDivdense/mul_26:z:0dense/Cast_1:y:0*
T0*
_output_shapes
:@M
dense/Neg_1Negdense/truediv_24:z:0*
T0*
_output_shapes
:@Q
dense/Round_6Rounddense/truediv_24:z:0*
T0*
_output_shapes
:@^
dense/add_18AddV2dense/Neg_1:y:0dense/Round_6:y:0*
T0*
_output_shapes
:@[
dense/StopGradient_1StopGradientdense/add_18:z:0*
T0*
_output_shapes
:@o
dense/add_19AddV2dense/truediv_24:z:0dense/StopGradient_1:output:0*
T0*
_output_shapes
:@b
dense/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
dense/clip_by_value/MinimumMinimumdense/add_19:z:0&dense/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:@Z
dense/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á
dense/clip_by_valueMaximumdense/clip_by_value/Minimum:z:0dense/clip_by_value/y:output:0*
T0*
_output_shapes
:@c
dense/mul_27Muldense/Cast_1:y:0dense/clip_by_value:z:0*
T0*
_output_shapes
:@W
dense/truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ao
dense/truediv_25RealDivdense/mul_27:z:0dense/truediv_25/y:output:0*
T0*
_output_shapes
:@S
dense/mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
dense/mul_28Muldense/mul_28/x:output:0dense/truediv_25:z:0*
T0*
_output_shapes
:@r
dense/ReadVariableOp_2ReadVariableOpdense_readvariableop_1_resource*
_output_shapes
:@*
dtype0W
dense/Neg_2Negdense/ReadVariableOp_2:value:0*
T0*
_output_shapes
:@]
dense/add_20AddV2dense/Neg_2:y:0dense/mul_28:z:0*
T0*
_output_shapes
:@S
dense/mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?c
dense/mul_29Muldense/mul_29/x:output:0dense/add_20:z:0*
T0*
_output_shapes
:@[
dense/StopGradient_2StopGradientdense/mul_29:z:0*
T0*
_output_shapes
:@r
dense/ReadVariableOp_3ReadVariableOpdense_readvariableop_1_resource*
_output_shapes
:@*
dtype0y
dense/add_21AddV2dense/ReadVariableOp_3:value:0dense/StopGradient_2:output:0*
T0*
_output_shapes
:@t
dense/BiasAddBiasAdddense/MatMul:product:0dense/add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ·
 batch_normalization/moments/meanMeandense/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:@¿
-batch_normalization/moments/SquaredDifferenceSquaredDifferencedense/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ú
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<ª
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0½
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:@´
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@ü
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<®
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0Ã
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@º
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:­
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@¦
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0°
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@
#batch_normalization/batchnorm/mul_1Muldense/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¤
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0¬
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@®
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout/dropout/MulMul'batch_normalization/batchnorm/add_1:z:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
dropout/dropout/ShapeShape'batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¾
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_

re_lu/SignSigndropout/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
	re_lu/AbsAbsre_lu/Sign:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
re_lu/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
	re_lu/subSubre_lu/sub/x:output:0re_lu/Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
	re_lu/addAddV2re_lu/Sign:y:0re_lu/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_

re_lu/TanhTanhdropout/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
	re_lu/NegNegre_lu/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
re_lu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
	re_lu/mulMulre_lu/mul/x:output:0re_lu/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
re_lu/add_1AddV2re_lu/Neg:y:0re_lu/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
re_lu/StopGradientStopGradientre_lu/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
re_lu/add_2AddV2re_lu/Tanh:y:0re_lu/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@O
dense_1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :O
dense_1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : c
dense_1/PowPowdense_1/Pow/x:output:0dense_1/Pow/y:output:0*
T0*
_output_shapes
: U
dense_1/CastCastdense_1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: v
dense_1/ReadVariableOpReadVariableOpdense_1_readvariableop_resource*
_output_shapes

:@@*
dtype0u
dense_1/truedivRealDivdense_1/ReadVariableOp:value:0dense_1/Cast:y:0*
T0*
_output_shapes

:@@P
dense_1/AbsAbsdense_1/truediv:z:0*
T0*
_output_shapes

:@@g
dense_1/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/MaxMaxdense_1/Abs:y:0&dense_1/Max/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(R
dense_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @i
dense_1/mulMuldense_1/Max:output:0dense_1/mul/y:output:0*
T0*
_output_shapes

:@X
dense_1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `At
dense_1/truediv_1RealDivdense_1/mul:z:0dense_1/truediv_1/y:output:0*
T0*
_output_shapes

:@R
dense_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3l
dense_1/addAddV2dense_1/truediv_1:z:0dense_1/add/y:output:0*
T0*
_output_shapes

:@L
dense_1/LogLogdense_1/add:z:0*
T0*
_output_shapes

:@X
dense_1/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?t
dense_1/truediv_2RealDivdense_1/Log:y:0dense_1/truediv_2/y:output:0*
T0*
_output_shapes

:@V
dense_1/RoundRounddense_1/truediv_2:z:0*
T0*
_output_shapes

:@T
dense_1/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @j
dense_1/Pow_1Powdense_1/Pow_1/x:output:0dense_1/Round:y:0*
T0*
_output_shapes

:@R
dense_1/Abs_1Absdense_1/truediv:z:0*
T0*
_output_shapes

:@@k
dense_1/truediv_3RealDivdense_1/Abs_1:y:0dense_1/Pow_1:z:0*
T0*
_output_shapes

:@@T
dense_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?p
dense_1/add_1AddV2dense_1/truediv_3:z:0dense_1/add_1/y:output:0*
T0*
_output_shapes

:@@R
dense_1/FloorFloordense_1/add_1:z:0*
T0*
_output_shapes

:@@S
dense_1/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@i
dense_1/LessLessdense_1/Floor:y:0dense_1/Less/y:output:0*
T0*
_output_shapes

:@@R
dense_1/SignSigndense_1/truediv:z:0*
T0*
_output_shapes

:@@x
'dense_1/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   \
dense_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/ones_likeFill0dense_1/ones_like/Shape/shape_as_tensor:output:0 dense_1/ones_like/Const:output:0*
T0*
_output_shapes

:@@T
dense_1/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `As
dense_1/mul_1Muldense_1/ones_like:output:0dense_1/mul_1/y:output:0*
T0*
_output_shapes

:@@X
dense_1/truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @v
dense_1/truediv_4RealDivdense_1/mul_1:z:0dense_1/truediv_4/y:output:0*
T0*
_output_shapes

:@@
dense_1/SelectV2SelectV2dense_1/Less:z:0dense_1/Floor:y:0dense_1/truediv_4:z:0*
T0*
_output_shapes

:@@j
dense_1/mul_2Muldense_1/Sign:y:0dense_1/SelectV2:output:0*
T0*
_output_shapes

:@@e
dense_1/Mul_3Muldense_1/truediv:z:0dense_1/mul_2:z:0*
T0*
_output_shapes

:@@h
dense_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/MeanMeandense_1/Mul_3:z:0'dense_1/Mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(c
dense_1/Mul_4Muldense_1/mul_2:z:0dense_1/mul_2:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_1Meandense_1/Mul_4:z:0)dense_1/Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(T
dense_1/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_1/add_2AddV2dense_1/Mean_1:output:0dense_1/add_2/y:output:0*
T0*
_output_shapes

:@o
dense_1/truediv_5RealDivdense_1/Mean:output:0dense_1/add_2:z:0*
T0*
_output_shapes

:@T
dense_1/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3p
dense_1/add_3AddV2dense_1/truediv_5:z:0dense_1/add_3/y:output:0*
T0*
_output_shapes

:@P
dense_1/Log_1Logdense_1/add_3:z:0*
T0*
_output_shapes

:@X
dense_1/truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?v
dense_1/truediv_6RealDivdense_1/Log_1:y:0dense_1/truediv_6/y:output:0*
T0*
_output_shapes

:@X
dense_1/Round_1Rounddense_1/truediv_6:z:0*
T0*
_output_shapes

:@T
dense_1/Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_1/Pow_2Powdense_1/Pow_2/x:output:0dense_1/Round_1:y:0*
T0*
_output_shapes

:@R
dense_1/Abs_2Absdense_1/truediv:z:0*
T0*
_output_shapes

:@@k
dense_1/truediv_7RealDivdense_1/Abs_2:y:0dense_1/Pow_2:z:0*
T0*
_output_shapes

:@@T
dense_1/add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?p
dense_1/add_4AddV2dense_1/truediv_7:z:0dense_1/add_4/y:output:0*
T0*
_output_shapes

:@@T
dense_1/Floor_1Floordense_1/add_4:z:0*
T0*
_output_shapes

:@@U
dense_1/Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_1/Less_1Lessdense_1/Floor_1:y:0dense_1/Less_1/y:output:0*
T0*
_output_shapes

:@@T
dense_1/Sign_1Signdense_1/truediv:z:0*
T0*
_output_shapes

:@@z
)dense_1/ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   ^
dense_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/ones_like_1Fill2dense_1/ones_like_1/Shape/shape_as_tensor:output:0"dense_1/ones_like_1/Const:output:0*
T0*
_output_shapes

:@@T
dense_1/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Au
dense_1/mul_5Muldense_1/ones_like_1:output:0dense_1/mul_5/y:output:0*
T0*
_output_shapes

:@@X
dense_1/truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @v
dense_1/truediv_8RealDivdense_1/mul_5:z:0dense_1/truediv_8/y:output:0*
T0*
_output_shapes

:@@
dense_1/SelectV2_1SelectV2dense_1/Less_1:z:0dense_1/Floor_1:y:0dense_1/truediv_8:z:0*
T0*
_output_shapes

:@@n
dense_1/mul_6Muldense_1/Sign_1:y:0dense_1/SelectV2_1:output:0*
T0*
_output_shapes

:@@e
dense_1/Mul_7Muldense_1/truediv:z:0dense_1/mul_6:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_2Meandense_1/Mul_7:z:0)dense_1/Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(c
dense_1/Mul_8Muldense_1/mul_6:z:0dense_1/mul_6:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_3Meandense_1/Mul_8:z:0)dense_1/Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(T
dense_1/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_1/add_5AddV2dense_1/Mean_3:output:0dense_1/add_5/y:output:0*
T0*
_output_shapes

:@q
dense_1/truediv_9RealDivdense_1/Mean_2:output:0dense_1/add_5:z:0*
T0*
_output_shapes

:@T
dense_1/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3p
dense_1/add_6AddV2dense_1/truediv_9:z:0dense_1/add_6/y:output:0*
T0*
_output_shapes

:@P
dense_1/Log_2Logdense_1/add_6:z:0*
T0*
_output_shapes

:@Y
dense_1/truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_1/truediv_10RealDivdense_1/Log_2:y:0dense_1/truediv_10/y:output:0*
T0*
_output_shapes

:@Y
dense_1/Round_2Rounddense_1/truediv_10:z:0*
T0*
_output_shapes

:@T
dense_1/Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_1/Pow_3Powdense_1/Pow_3/x:output:0dense_1/Round_2:y:0*
T0*
_output_shapes

:@R
dense_1/Abs_3Absdense_1/truediv:z:0*
T0*
_output_shapes

:@@l
dense_1/truediv_11RealDivdense_1/Abs_3:y:0dense_1/Pow_3:z:0*
T0*
_output_shapes

:@@T
dense_1/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?q
dense_1/add_7AddV2dense_1/truediv_11:z:0dense_1/add_7/y:output:0*
T0*
_output_shapes

:@@T
dense_1/Floor_2Floordense_1/add_7:z:0*
T0*
_output_shapes

:@@U
dense_1/Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_1/Less_2Lessdense_1/Floor_2:y:0dense_1/Less_2/y:output:0*
T0*
_output_shapes

:@@T
dense_1/Sign_2Signdense_1/truediv:z:0*
T0*
_output_shapes

:@@z
)dense_1/ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   ^
dense_1/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/ones_like_2Fill2dense_1/ones_like_2/Shape/shape_as_tensor:output:0"dense_1/ones_like_2/Const:output:0*
T0*
_output_shapes

:@@T
dense_1/mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Au
dense_1/mul_9Muldense_1/ones_like_2:output:0dense_1/mul_9/y:output:0*
T0*
_output_shapes

:@@Y
dense_1/truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @x
dense_1/truediv_12RealDivdense_1/mul_9:z:0dense_1/truediv_12/y:output:0*
T0*
_output_shapes

:@@
dense_1/SelectV2_2SelectV2dense_1/Less_2:z:0dense_1/Floor_2:y:0dense_1/truediv_12:z:0*
T0*
_output_shapes

:@@o
dense_1/mul_10Muldense_1/Sign_2:y:0dense_1/SelectV2_2:output:0*
T0*
_output_shapes

:@@g
dense_1/Mul_11Muldense_1/truediv:z:0dense_1/mul_10:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_4Meandense_1/Mul_11:z:0)dense_1/Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(f
dense_1/Mul_12Muldense_1/mul_10:z:0dense_1/mul_10:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_5Meandense_1/Mul_12:z:0)dense_1/Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(T
dense_1/add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_1/add_8AddV2dense_1/Mean_5:output:0dense_1/add_8/y:output:0*
T0*
_output_shapes

:@r
dense_1/truediv_13RealDivdense_1/Mean_4:output:0dense_1/add_8:z:0*
T0*
_output_shapes

:@T
dense_1/add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3q
dense_1/add_9AddV2dense_1/truediv_13:z:0dense_1/add_9/y:output:0*
T0*
_output_shapes

:@P
dense_1/Log_3Logdense_1/add_9:z:0*
T0*
_output_shapes

:@Y
dense_1/truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_1/truediv_14RealDivdense_1/Log_3:y:0dense_1/truediv_14/y:output:0*
T0*
_output_shapes

:@Y
dense_1/Round_3Rounddense_1/truediv_14:z:0*
T0*
_output_shapes

:@T
dense_1/Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_1/Pow_4Powdense_1/Pow_4/x:output:0dense_1/Round_3:y:0*
T0*
_output_shapes

:@R
dense_1/Abs_4Absdense_1/truediv:z:0*
T0*
_output_shapes

:@@l
dense_1/truediv_15RealDivdense_1/Abs_4:y:0dense_1/Pow_4:z:0*
T0*
_output_shapes

:@@U
dense_1/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?s
dense_1/add_10AddV2dense_1/truediv_15:z:0dense_1/add_10/y:output:0*
T0*
_output_shapes

:@@U
dense_1/Floor_3Floordense_1/add_10:z:0*
T0*
_output_shapes

:@@U
dense_1/Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_1/Less_3Lessdense_1/Floor_3:y:0dense_1/Less_3/y:output:0*
T0*
_output_shapes

:@@T
dense_1/Sign_3Signdense_1/truediv:z:0*
T0*
_output_shapes

:@@z
)dense_1/ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   ^
dense_1/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/ones_like_3Fill2dense_1/ones_like_3/Shape/shape_as_tensor:output:0"dense_1/ones_like_3/Const:output:0*
T0*
_output_shapes

:@@U
dense_1/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Aw
dense_1/mul_13Muldense_1/ones_like_3:output:0dense_1/mul_13/y:output:0*
T0*
_output_shapes

:@@Y
dense_1/truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @y
dense_1/truediv_16RealDivdense_1/mul_13:z:0dense_1/truediv_16/y:output:0*
T0*
_output_shapes

:@@
dense_1/SelectV2_3SelectV2dense_1/Less_3:z:0dense_1/Floor_3:y:0dense_1/truediv_16:z:0*
T0*
_output_shapes

:@@o
dense_1/mul_14Muldense_1/Sign_3:y:0dense_1/SelectV2_3:output:0*
T0*
_output_shapes

:@@g
dense_1/Mul_15Muldense_1/truediv:z:0dense_1/mul_14:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_6Meandense_1/Mul_15:z:0)dense_1/Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(f
dense_1/Mul_16Muldense_1/mul_14:z:0dense_1/mul_14:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_7Meandense_1/Mul_16:z:0)dense_1/Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(U
dense_1/add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3t
dense_1/add_11AddV2dense_1/Mean_7:output:0dense_1/add_11/y:output:0*
T0*
_output_shapes

:@s
dense_1/truediv_17RealDivdense_1/Mean_6:output:0dense_1/add_11:z:0*
T0*
_output_shapes

:@U
dense_1/add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3s
dense_1/add_12AddV2dense_1/truediv_17:z:0dense_1/add_12/y:output:0*
T0*
_output_shapes

:@Q
dense_1/Log_4Logdense_1/add_12:z:0*
T0*
_output_shapes

:@Y
dense_1/truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_1/truediv_18RealDivdense_1/Log_4:y:0dense_1/truediv_18/y:output:0*
T0*
_output_shapes

:@Y
dense_1/Round_4Rounddense_1/truediv_18:z:0*
T0*
_output_shapes

:@T
dense_1/Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_1/Pow_5Powdense_1/Pow_5/x:output:0dense_1/Round_4:y:0*
T0*
_output_shapes

:@R
dense_1/Abs_5Absdense_1/truediv:z:0*
T0*
_output_shapes

:@@l
dense_1/truediv_19RealDivdense_1/Abs_5:y:0dense_1/Pow_5:z:0*
T0*
_output_shapes

:@@U
dense_1/add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?s
dense_1/add_13AddV2dense_1/truediv_19:z:0dense_1/add_13/y:output:0*
T0*
_output_shapes

:@@U
dense_1/Floor_4Floordense_1/add_13:z:0*
T0*
_output_shapes

:@@U
dense_1/Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_1/Less_4Lessdense_1/Floor_4:y:0dense_1/Less_4/y:output:0*
T0*
_output_shapes

:@@T
dense_1/Sign_4Signdense_1/truediv:z:0*
T0*
_output_shapes

:@@z
)dense_1/ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   ^
dense_1/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/ones_like_4Fill2dense_1/ones_like_4/Shape/shape_as_tensor:output:0"dense_1/ones_like_4/Const:output:0*
T0*
_output_shapes

:@@U
dense_1/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Aw
dense_1/mul_17Muldense_1/ones_like_4:output:0dense_1/mul_17/y:output:0*
T0*
_output_shapes

:@@Y
dense_1/truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @y
dense_1/truediv_20RealDivdense_1/mul_17:z:0dense_1/truediv_20/y:output:0*
T0*
_output_shapes

:@@
dense_1/SelectV2_4SelectV2dense_1/Less_4:z:0dense_1/Floor_4:y:0dense_1/truediv_20:z:0*
T0*
_output_shapes

:@@o
dense_1/mul_18Muldense_1/Sign_4:y:0dense_1/SelectV2_4:output:0*
T0*
_output_shapes

:@@g
dense_1/Mul_19Muldense_1/truediv:z:0dense_1/mul_18:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_8Meandense_1/Mul_19:z:0)dense_1/Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(f
dense_1/Mul_20Muldense_1/mul_18:z:0dense_1/mul_18:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_9Meandense_1/Mul_20:z:0)dense_1/Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(U
dense_1/add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3t
dense_1/add_14AddV2dense_1/Mean_9:output:0dense_1/add_14/y:output:0*
T0*
_output_shapes

:@s
dense_1/truediv_21RealDivdense_1/Mean_8:output:0dense_1/add_14:z:0*
T0*
_output_shapes

:@U
dense_1/add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3s
dense_1/add_15AddV2dense_1/truediv_21:z:0dense_1/add_15/y:output:0*
T0*
_output_shapes

:@Q
dense_1/Log_5Logdense_1/add_15:z:0*
T0*
_output_shapes

:@Y
dense_1/truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_1/truediv_22RealDivdense_1/Log_5:y:0dense_1/truediv_22/y:output:0*
T0*
_output_shapes

:@Y
dense_1/Round_5Rounddense_1/truediv_22:z:0*
T0*
_output_shapes

:@T
dense_1/Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_1/Pow_6Powdense_1/Pow_6/x:output:0dense_1/Round_5:y:0*
T0*
_output_shapes

:@U
dense_1/mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Al
dense_1/mul_21Muldense_1/Pow_6:z:0dense_1/mul_21/y:output:0*
T0*
_output_shapes

:@e
dense_1/mul_22Muldense_1/Cast:y:0dense_1/truediv:z:0*
T0*
_output_shapes

:@@d
dense_1/mul_23Muldense_1/Cast:y:0dense_1/mul_18:z:0*
T0*
_output_shapes

:@@Y
dense_1/truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ay
dense_1/truediv_23RealDivdense_1/mul_23:z:0dense_1/truediv_23/y:output:0*
T0*
_output_shapes

:@@j
dense_1/mul_24Muldense_1/mul_21:z:0dense_1/truediv_23:z:0*
T0*
_output_shapes

:@@O
dense_1/NegNegdense_1/mul_22:z:0*
T0*
_output_shapes

:@@e
dense_1/add_16AddV2dense_1/Neg:y:0dense_1/mul_24:z:0*
T0*
_output_shapes

:@@U
dense_1/mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
dense_1/mul_25Muldense_1/mul_25/x:output:0dense_1/add_16:z:0*
T0*
_output_shapes

:@@a
dense_1/StopGradientStopGradientdense_1/mul_25:z:0*
T0*
_output_shapes

:@@s
dense_1/add_17AddV2dense_1/mul_22:z:0dense_1/StopGradient:output:0*
T0*
_output_shapes

:@@o
dense_1/MatMulMatMulre_lu/add_2:z:0dense_1/add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
dense_1/Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :Q
dense_1/Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : i
dense_1/Pow_7Powdense_1/Pow_7/x:output:0dense_1/Pow_7/y:output:0*
T0*
_output_shapes
: Y
dense_1/Cast_1Castdense_1/Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: v
dense_1/ReadVariableOp_1ReadVariableOp!dense_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0U
dense_1/mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aw
dense_1/mul_26Mul dense_1/ReadVariableOp_1:value:0dense_1/mul_26/y:output:0*
T0*
_output_shapes
:@j
dense_1/truediv_24RealDivdense_1/mul_26:z:0dense_1/Cast_1:y:0*
T0*
_output_shapes
:@Q
dense_1/Neg_1Negdense_1/truediv_24:z:0*
T0*
_output_shapes
:@U
dense_1/Round_6Rounddense_1/truediv_24:z:0*
T0*
_output_shapes
:@d
dense_1/add_18AddV2dense_1/Neg_1:y:0dense_1/Round_6:y:0*
T0*
_output_shapes
:@_
dense_1/StopGradient_1StopGradientdense_1/add_18:z:0*
T0*
_output_shapes
:@u
dense_1/add_19AddV2dense_1/truediv_24:z:0dense_1/StopGradient_1:output:0*
T0*
_output_shapes
:@d
dense_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
dense_1/clip_by_value/MinimumMinimumdense_1/add_19:z:0(dense_1/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:@\
dense_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á
dense_1/clip_by_valueMaximum!dense_1/clip_by_value/Minimum:z:0 dense_1/clip_by_value/y:output:0*
T0*
_output_shapes
:@i
dense_1/mul_27Muldense_1/Cast_1:y:0dense_1/clip_by_value:z:0*
T0*
_output_shapes
:@Y
dense_1/truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Au
dense_1/truediv_25RealDivdense_1/mul_27:z:0dense_1/truediv_25/y:output:0*
T0*
_output_shapes
:@U
dense_1/mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
dense_1/mul_28Muldense_1/mul_28/x:output:0dense_1/truediv_25:z:0*
T0*
_output_shapes
:@v
dense_1/ReadVariableOp_2ReadVariableOp!dense_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0[
dense_1/Neg_2Neg dense_1/ReadVariableOp_2:value:0*
T0*
_output_shapes
:@c
dense_1/add_20AddV2dense_1/Neg_2:y:0dense_1/mul_28:z:0*
T0*
_output_shapes
:@U
dense_1/mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
dense_1/mul_29Muldense_1/mul_29/x:output:0dense_1/add_20:z:0*
T0*
_output_shapes
:@_
dense_1/StopGradient_2StopGradientdense_1/mul_29:z:0*
T0*
_output_shapes
:@v
dense_1/ReadVariableOp_3ReadVariableOp!dense_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0
dense_1/add_21AddV2 dense_1/ReadVariableOp_3:value:0dense_1/StopGradient_2:output:0*
T0*
_output_shapes
:@z
dense_1/BiasAddBiasAdddense_1/MatMul:product:0dense_1/add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ½
"batch_normalization_1/moments/meanMeandense_1/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes

:@Å
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: à
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 p
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<®
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0Ã
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:@º
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0É
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@À
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:³
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@ª
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0¶
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@¡
%batch_normalization_1/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ª
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@¢
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0²
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@´
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_1/dropout/MulMul)batch_normalization_1/batchnorm/add_1:z:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
dropout_1/dropout/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
: 
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ä
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
re_lu_1/SignSigndropout_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@V
re_lu_1/AbsAbsre_lu_1/Sign:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
re_lu_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
re_lu_1/subSubre_lu_1/sub/x:output:0re_lu_1/Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
re_lu_1/addAddV2re_lu_1/Sign:y:0re_lu_1/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
re_lu_1/TanhTanhdropout_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@V
re_lu_1/NegNegre_lu_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
re_lu_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
re_lu_1/mulMulre_lu_1/mul/x:output:0re_lu_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
re_lu_1/add_1AddV2re_lu_1/Neg:y:0re_lu_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
re_lu_1/StopGradientStopGradientre_lu_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
re_lu_1/add_2AddV2re_lu_1/Tanh:y:0re_lu_1/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@O
dense_2/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :O
dense_2/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : c
dense_2/PowPowdense_2/Pow/x:output:0dense_2/Pow/y:output:0*
T0*
_output_shapes
: U
dense_2/CastCastdense_2/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: v
dense_2/ReadVariableOpReadVariableOpdense_2_readvariableop_resource*
_output_shapes

:@@*
dtype0u
dense_2/truedivRealDivdense_2/ReadVariableOp:value:0dense_2/Cast:y:0*
T0*
_output_shapes

:@@P
dense_2/AbsAbsdense_2/truediv:z:0*
T0*
_output_shapes

:@@g
dense_2/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/MaxMaxdense_2/Abs:y:0&dense_2/Max/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(R
dense_2/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @i
dense_2/mulMuldense_2/Max:output:0dense_2/mul/y:output:0*
T0*
_output_shapes

:@X
dense_2/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `At
dense_2/truediv_1RealDivdense_2/mul:z:0dense_2/truediv_1/y:output:0*
T0*
_output_shapes

:@R
dense_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3l
dense_2/addAddV2dense_2/truediv_1:z:0dense_2/add/y:output:0*
T0*
_output_shapes

:@L
dense_2/LogLogdense_2/add:z:0*
T0*
_output_shapes

:@X
dense_2/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?t
dense_2/truediv_2RealDivdense_2/Log:y:0dense_2/truediv_2/y:output:0*
T0*
_output_shapes

:@V
dense_2/RoundRounddense_2/truediv_2:z:0*
T0*
_output_shapes

:@T
dense_2/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @j
dense_2/Pow_1Powdense_2/Pow_1/x:output:0dense_2/Round:y:0*
T0*
_output_shapes

:@R
dense_2/Abs_1Absdense_2/truediv:z:0*
T0*
_output_shapes

:@@k
dense_2/truediv_3RealDivdense_2/Abs_1:y:0dense_2/Pow_1:z:0*
T0*
_output_shapes

:@@T
dense_2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?p
dense_2/add_1AddV2dense_2/truediv_3:z:0dense_2/add_1/y:output:0*
T0*
_output_shapes

:@@R
dense_2/FloorFloordense_2/add_1:z:0*
T0*
_output_shapes

:@@S
dense_2/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@i
dense_2/LessLessdense_2/Floor:y:0dense_2/Less/y:output:0*
T0*
_output_shapes

:@@R
dense_2/SignSigndense_2/truediv:z:0*
T0*
_output_shapes

:@@x
'dense_2/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   \
dense_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_2/ones_likeFill0dense_2/ones_like/Shape/shape_as_tensor:output:0 dense_2/ones_like/Const:output:0*
T0*
_output_shapes

:@@T
dense_2/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `As
dense_2/mul_1Muldense_2/ones_like:output:0dense_2/mul_1/y:output:0*
T0*
_output_shapes

:@@X
dense_2/truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @v
dense_2/truediv_4RealDivdense_2/mul_1:z:0dense_2/truediv_4/y:output:0*
T0*
_output_shapes

:@@
dense_2/SelectV2SelectV2dense_2/Less:z:0dense_2/Floor:y:0dense_2/truediv_4:z:0*
T0*
_output_shapes

:@@j
dense_2/mul_2Muldense_2/Sign:y:0dense_2/SelectV2:output:0*
T0*
_output_shapes

:@@e
dense_2/Mul_3Muldense_2/truediv:z:0dense_2/mul_2:z:0*
T0*
_output_shapes

:@@h
dense_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/MeanMeandense_2/Mul_3:z:0'dense_2/Mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(c
dense_2/Mul_4Muldense_2/mul_2:z:0dense_2/mul_2:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_1Meandense_2/Mul_4:z:0)dense_2/Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(T
dense_2/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_2/add_2AddV2dense_2/Mean_1:output:0dense_2/add_2/y:output:0*
T0*
_output_shapes

:@o
dense_2/truediv_5RealDivdense_2/Mean:output:0dense_2/add_2:z:0*
T0*
_output_shapes

:@T
dense_2/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3p
dense_2/add_3AddV2dense_2/truediv_5:z:0dense_2/add_3/y:output:0*
T0*
_output_shapes

:@P
dense_2/Log_1Logdense_2/add_3:z:0*
T0*
_output_shapes

:@X
dense_2/truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?v
dense_2/truediv_6RealDivdense_2/Log_1:y:0dense_2/truediv_6/y:output:0*
T0*
_output_shapes

:@X
dense_2/Round_1Rounddense_2/truediv_6:z:0*
T0*
_output_shapes

:@T
dense_2/Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_2/Pow_2Powdense_2/Pow_2/x:output:0dense_2/Round_1:y:0*
T0*
_output_shapes

:@R
dense_2/Abs_2Absdense_2/truediv:z:0*
T0*
_output_shapes

:@@k
dense_2/truediv_7RealDivdense_2/Abs_2:y:0dense_2/Pow_2:z:0*
T0*
_output_shapes

:@@T
dense_2/add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?p
dense_2/add_4AddV2dense_2/truediv_7:z:0dense_2/add_4/y:output:0*
T0*
_output_shapes

:@@T
dense_2/Floor_1Floordense_2/add_4:z:0*
T0*
_output_shapes

:@@U
dense_2/Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_2/Less_1Lessdense_2/Floor_1:y:0dense_2/Less_1/y:output:0*
T0*
_output_shapes

:@@T
dense_2/Sign_1Signdense_2/truediv:z:0*
T0*
_output_shapes

:@@z
)dense_2/ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   ^
dense_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_2/ones_like_1Fill2dense_2/ones_like_1/Shape/shape_as_tensor:output:0"dense_2/ones_like_1/Const:output:0*
T0*
_output_shapes

:@@T
dense_2/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Au
dense_2/mul_5Muldense_2/ones_like_1:output:0dense_2/mul_5/y:output:0*
T0*
_output_shapes

:@@X
dense_2/truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @v
dense_2/truediv_8RealDivdense_2/mul_5:z:0dense_2/truediv_8/y:output:0*
T0*
_output_shapes

:@@
dense_2/SelectV2_1SelectV2dense_2/Less_1:z:0dense_2/Floor_1:y:0dense_2/truediv_8:z:0*
T0*
_output_shapes

:@@n
dense_2/mul_6Muldense_2/Sign_1:y:0dense_2/SelectV2_1:output:0*
T0*
_output_shapes

:@@e
dense_2/Mul_7Muldense_2/truediv:z:0dense_2/mul_6:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_2Meandense_2/Mul_7:z:0)dense_2/Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(c
dense_2/Mul_8Muldense_2/mul_6:z:0dense_2/mul_6:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_3Meandense_2/Mul_8:z:0)dense_2/Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(T
dense_2/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_2/add_5AddV2dense_2/Mean_3:output:0dense_2/add_5/y:output:0*
T0*
_output_shapes

:@q
dense_2/truediv_9RealDivdense_2/Mean_2:output:0dense_2/add_5:z:0*
T0*
_output_shapes

:@T
dense_2/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3p
dense_2/add_6AddV2dense_2/truediv_9:z:0dense_2/add_6/y:output:0*
T0*
_output_shapes

:@P
dense_2/Log_2Logdense_2/add_6:z:0*
T0*
_output_shapes

:@Y
dense_2/truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_2/truediv_10RealDivdense_2/Log_2:y:0dense_2/truediv_10/y:output:0*
T0*
_output_shapes

:@Y
dense_2/Round_2Rounddense_2/truediv_10:z:0*
T0*
_output_shapes

:@T
dense_2/Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_2/Pow_3Powdense_2/Pow_3/x:output:0dense_2/Round_2:y:0*
T0*
_output_shapes

:@R
dense_2/Abs_3Absdense_2/truediv:z:0*
T0*
_output_shapes

:@@l
dense_2/truediv_11RealDivdense_2/Abs_3:y:0dense_2/Pow_3:z:0*
T0*
_output_shapes

:@@T
dense_2/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?q
dense_2/add_7AddV2dense_2/truediv_11:z:0dense_2/add_7/y:output:0*
T0*
_output_shapes

:@@T
dense_2/Floor_2Floordense_2/add_7:z:0*
T0*
_output_shapes

:@@U
dense_2/Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_2/Less_2Lessdense_2/Floor_2:y:0dense_2/Less_2/y:output:0*
T0*
_output_shapes

:@@T
dense_2/Sign_2Signdense_2/truediv:z:0*
T0*
_output_shapes

:@@z
)dense_2/ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   ^
dense_2/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_2/ones_like_2Fill2dense_2/ones_like_2/Shape/shape_as_tensor:output:0"dense_2/ones_like_2/Const:output:0*
T0*
_output_shapes

:@@T
dense_2/mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Au
dense_2/mul_9Muldense_2/ones_like_2:output:0dense_2/mul_9/y:output:0*
T0*
_output_shapes

:@@Y
dense_2/truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @x
dense_2/truediv_12RealDivdense_2/mul_9:z:0dense_2/truediv_12/y:output:0*
T0*
_output_shapes

:@@
dense_2/SelectV2_2SelectV2dense_2/Less_2:z:0dense_2/Floor_2:y:0dense_2/truediv_12:z:0*
T0*
_output_shapes

:@@o
dense_2/mul_10Muldense_2/Sign_2:y:0dense_2/SelectV2_2:output:0*
T0*
_output_shapes

:@@g
dense_2/Mul_11Muldense_2/truediv:z:0dense_2/mul_10:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_4Meandense_2/Mul_11:z:0)dense_2/Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(f
dense_2/Mul_12Muldense_2/mul_10:z:0dense_2/mul_10:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_5Meandense_2/Mul_12:z:0)dense_2/Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(T
dense_2/add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_2/add_8AddV2dense_2/Mean_5:output:0dense_2/add_8/y:output:0*
T0*
_output_shapes

:@r
dense_2/truediv_13RealDivdense_2/Mean_4:output:0dense_2/add_8:z:0*
T0*
_output_shapes

:@T
dense_2/add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3q
dense_2/add_9AddV2dense_2/truediv_13:z:0dense_2/add_9/y:output:0*
T0*
_output_shapes

:@P
dense_2/Log_3Logdense_2/add_9:z:0*
T0*
_output_shapes

:@Y
dense_2/truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_2/truediv_14RealDivdense_2/Log_3:y:0dense_2/truediv_14/y:output:0*
T0*
_output_shapes

:@Y
dense_2/Round_3Rounddense_2/truediv_14:z:0*
T0*
_output_shapes

:@T
dense_2/Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_2/Pow_4Powdense_2/Pow_4/x:output:0dense_2/Round_3:y:0*
T0*
_output_shapes

:@R
dense_2/Abs_4Absdense_2/truediv:z:0*
T0*
_output_shapes

:@@l
dense_2/truediv_15RealDivdense_2/Abs_4:y:0dense_2/Pow_4:z:0*
T0*
_output_shapes

:@@U
dense_2/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?s
dense_2/add_10AddV2dense_2/truediv_15:z:0dense_2/add_10/y:output:0*
T0*
_output_shapes

:@@U
dense_2/Floor_3Floordense_2/add_10:z:0*
T0*
_output_shapes

:@@U
dense_2/Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_2/Less_3Lessdense_2/Floor_3:y:0dense_2/Less_3/y:output:0*
T0*
_output_shapes

:@@T
dense_2/Sign_3Signdense_2/truediv:z:0*
T0*
_output_shapes

:@@z
)dense_2/ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   ^
dense_2/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_2/ones_like_3Fill2dense_2/ones_like_3/Shape/shape_as_tensor:output:0"dense_2/ones_like_3/Const:output:0*
T0*
_output_shapes

:@@U
dense_2/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Aw
dense_2/mul_13Muldense_2/ones_like_3:output:0dense_2/mul_13/y:output:0*
T0*
_output_shapes

:@@Y
dense_2/truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @y
dense_2/truediv_16RealDivdense_2/mul_13:z:0dense_2/truediv_16/y:output:0*
T0*
_output_shapes

:@@
dense_2/SelectV2_3SelectV2dense_2/Less_3:z:0dense_2/Floor_3:y:0dense_2/truediv_16:z:0*
T0*
_output_shapes

:@@o
dense_2/mul_14Muldense_2/Sign_3:y:0dense_2/SelectV2_3:output:0*
T0*
_output_shapes

:@@g
dense_2/Mul_15Muldense_2/truediv:z:0dense_2/mul_14:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_6Meandense_2/Mul_15:z:0)dense_2/Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(f
dense_2/Mul_16Muldense_2/mul_14:z:0dense_2/mul_14:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_7Meandense_2/Mul_16:z:0)dense_2/Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(U
dense_2/add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3t
dense_2/add_11AddV2dense_2/Mean_7:output:0dense_2/add_11/y:output:0*
T0*
_output_shapes

:@s
dense_2/truediv_17RealDivdense_2/Mean_6:output:0dense_2/add_11:z:0*
T0*
_output_shapes

:@U
dense_2/add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3s
dense_2/add_12AddV2dense_2/truediv_17:z:0dense_2/add_12/y:output:0*
T0*
_output_shapes

:@Q
dense_2/Log_4Logdense_2/add_12:z:0*
T0*
_output_shapes

:@Y
dense_2/truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_2/truediv_18RealDivdense_2/Log_4:y:0dense_2/truediv_18/y:output:0*
T0*
_output_shapes

:@Y
dense_2/Round_4Rounddense_2/truediv_18:z:0*
T0*
_output_shapes

:@T
dense_2/Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_2/Pow_5Powdense_2/Pow_5/x:output:0dense_2/Round_4:y:0*
T0*
_output_shapes

:@R
dense_2/Abs_5Absdense_2/truediv:z:0*
T0*
_output_shapes

:@@l
dense_2/truediv_19RealDivdense_2/Abs_5:y:0dense_2/Pow_5:z:0*
T0*
_output_shapes

:@@U
dense_2/add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?s
dense_2/add_13AddV2dense_2/truediv_19:z:0dense_2/add_13/y:output:0*
T0*
_output_shapes

:@@U
dense_2/Floor_4Floordense_2/add_13:z:0*
T0*
_output_shapes

:@@U
dense_2/Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_2/Less_4Lessdense_2/Floor_4:y:0dense_2/Less_4/y:output:0*
T0*
_output_shapes

:@@T
dense_2/Sign_4Signdense_2/truediv:z:0*
T0*
_output_shapes

:@@z
)dense_2/ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   ^
dense_2/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_2/ones_like_4Fill2dense_2/ones_like_4/Shape/shape_as_tensor:output:0"dense_2/ones_like_4/Const:output:0*
T0*
_output_shapes

:@@U
dense_2/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Aw
dense_2/mul_17Muldense_2/ones_like_4:output:0dense_2/mul_17/y:output:0*
T0*
_output_shapes

:@@Y
dense_2/truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @y
dense_2/truediv_20RealDivdense_2/mul_17:z:0dense_2/truediv_20/y:output:0*
T0*
_output_shapes

:@@
dense_2/SelectV2_4SelectV2dense_2/Less_4:z:0dense_2/Floor_4:y:0dense_2/truediv_20:z:0*
T0*
_output_shapes

:@@o
dense_2/mul_18Muldense_2/Sign_4:y:0dense_2/SelectV2_4:output:0*
T0*
_output_shapes

:@@g
dense_2/Mul_19Muldense_2/truediv:z:0dense_2/mul_18:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_8Meandense_2/Mul_19:z:0)dense_2/Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(f
dense_2/Mul_20Muldense_2/mul_18:z:0dense_2/mul_18:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_9Meandense_2/Mul_20:z:0)dense_2/Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(U
dense_2/add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3t
dense_2/add_14AddV2dense_2/Mean_9:output:0dense_2/add_14/y:output:0*
T0*
_output_shapes

:@s
dense_2/truediv_21RealDivdense_2/Mean_8:output:0dense_2/add_14:z:0*
T0*
_output_shapes

:@U
dense_2/add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3s
dense_2/add_15AddV2dense_2/truediv_21:z:0dense_2/add_15/y:output:0*
T0*
_output_shapes

:@Q
dense_2/Log_5Logdense_2/add_15:z:0*
T0*
_output_shapes

:@Y
dense_2/truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_2/truediv_22RealDivdense_2/Log_5:y:0dense_2/truediv_22/y:output:0*
T0*
_output_shapes

:@Y
dense_2/Round_5Rounddense_2/truediv_22:z:0*
T0*
_output_shapes

:@T
dense_2/Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_2/Pow_6Powdense_2/Pow_6/x:output:0dense_2/Round_5:y:0*
T0*
_output_shapes

:@U
dense_2/mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Al
dense_2/mul_21Muldense_2/Pow_6:z:0dense_2/mul_21/y:output:0*
T0*
_output_shapes

:@e
dense_2/mul_22Muldense_2/Cast:y:0dense_2/truediv:z:0*
T0*
_output_shapes

:@@d
dense_2/mul_23Muldense_2/Cast:y:0dense_2/mul_18:z:0*
T0*
_output_shapes

:@@Y
dense_2/truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ay
dense_2/truediv_23RealDivdense_2/mul_23:z:0dense_2/truediv_23/y:output:0*
T0*
_output_shapes

:@@j
dense_2/mul_24Muldense_2/mul_21:z:0dense_2/truediv_23:z:0*
T0*
_output_shapes

:@@O
dense_2/NegNegdense_2/mul_22:z:0*
T0*
_output_shapes

:@@e
dense_2/add_16AddV2dense_2/Neg:y:0dense_2/mul_24:z:0*
T0*
_output_shapes

:@@U
dense_2/mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
dense_2/mul_25Muldense_2/mul_25/x:output:0dense_2/add_16:z:0*
T0*
_output_shapes

:@@a
dense_2/StopGradientStopGradientdense_2/mul_25:z:0*
T0*
_output_shapes

:@@s
dense_2/add_17AddV2dense_2/mul_22:z:0dense_2/StopGradient:output:0*
T0*
_output_shapes

:@@q
dense_2/MatMulMatMulre_lu_1/add_2:z:0dense_2/add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
dense_2/Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :Q
dense_2/Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : i
dense_2/Pow_7Powdense_2/Pow_7/x:output:0dense_2/Pow_7/y:output:0*
T0*
_output_shapes
: Y
dense_2/Cast_1Castdense_2/Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: v
dense_2/ReadVariableOp_1ReadVariableOp!dense_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0U
dense_2/mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aw
dense_2/mul_26Mul dense_2/ReadVariableOp_1:value:0dense_2/mul_26/y:output:0*
T0*
_output_shapes
:@j
dense_2/truediv_24RealDivdense_2/mul_26:z:0dense_2/Cast_1:y:0*
T0*
_output_shapes
:@Q
dense_2/Neg_1Negdense_2/truediv_24:z:0*
T0*
_output_shapes
:@U
dense_2/Round_6Rounddense_2/truediv_24:z:0*
T0*
_output_shapes
:@d
dense_2/add_18AddV2dense_2/Neg_1:y:0dense_2/Round_6:y:0*
T0*
_output_shapes
:@_
dense_2/StopGradient_1StopGradientdense_2/add_18:z:0*
T0*
_output_shapes
:@u
dense_2/add_19AddV2dense_2/truediv_24:z:0dense_2/StopGradient_1:output:0*
T0*
_output_shapes
:@d
dense_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
dense_2/clip_by_value/MinimumMinimumdense_2/add_19:z:0(dense_2/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:@\
dense_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á
dense_2/clip_by_valueMaximum!dense_2/clip_by_value/Minimum:z:0 dense_2/clip_by_value/y:output:0*
T0*
_output_shapes
:@i
dense_2/mul_27Muldense_2/Cast_1:y:0dense_2/clip_by_value:z:0*
T0*
_output_shapes
:@Y
dense_2/truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Au
dense_2/truediv_25RealDivdense_2/mul_27:z:0dense_2/truediv_25/y:output:0*
T0*
_output_shapes
:@U
dense_2/mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
dense_2/mul_28Muldense_2/mul_28/x:output:0dense_2/truediv_25:z:0*
T0*
_output_shapes
:@v
dense_2/ReadVariableOp_2ReadVariableOp!dense_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0[
dense_2/Neg_2Neg dense_2/ReadVariableOp_2:value:0*
T0*
_output_shapes
:@c
dense_2/add_20AddV2dense_2/Neg_2:y:0dense_2/mul_28:z:0*
T0*
_output_shapes
:@U
dense_2/mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
dense_2/mul_29Muldense_2/mul_29/x:output:0dense_2/add_20:z:0*
T0*
_output_shapes
:@_
dense_2/StopGradient_2StopGradientdense_2/mul_29:z:0*
T0*
_output_shapes
:@v
dense_2/ReadVariableOp_3ReadVariableOp!dense_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0
dense_2/add_21AddV2 dense_2/ReadVariableOp_3:value:0dense_2/StopGradient_2:output:0*
T0*
_output_shapes
:@z
dense_2/BiasAddBiasAdddense_2/MatMul:product:0dense_2/add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ½
"batch_normalization_2/moments/meanMeandense_2/BiasAdd:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:@Å
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_2/BiasAdd:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: à
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 p
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<®
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0Ã
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:@º
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0É
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@À
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:³
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@ª
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0¶
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@¡
%batch_normalization_2/batchnorm/mul_1Muldense_2/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ª
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@¢
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0²
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@´
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_2/dropout/MulMul)batch_normalization_2/batchnorm/add_1:z:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
dropout_2/dropout/ShapeShape)batch_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
: 
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ä
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
re_lu_2/SignSigndropout_2/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@V
re_lu_2/AbsAbsre_lu_2/Sign:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
re_lu_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
re_lu_2/subSubre_lu_2/sub/x:output:0re_lu_2/Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
re_lu_2/addAddV2re_lu_2/Sign:y:0re_lu_2/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
re_lu_2/TanhTanhdropout_2/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@V
re_lu_2/NegNegre_lu_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
re_lu_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
re_lu_2/mulMulre_lu_2/mul/x:output:0re_lu_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
re_lu_2/add_1AddV2re_lu_2/Neg:y:0re_lu_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
re_lu_2/StopGradientStopGradientre_lu_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
re_lu_2/add_2AddV2re_lu_2/Tanh:y:0re_lu_2/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@O
dense_3/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :O
dense_3/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : c
dense_3/PowPowdense_3/Pow/x:output:0dense_3/Pow/y:output:0*
T0*
_output_shapes
: U
dense_3/CastCastdense_3/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: v
dense_3/ReadVariableOpReadVariableOpdense_3_readvariableop_resource*
_output_shapes

:@*
dtype0u
dense_3/truedivRealDivdense_3/ReadVariableOp:value:0dense_3/Cast:y:0*
T0*
_output_shapes

:@P
dense_3/AbsAbsdense_3/truediv:z:0*
T0*
_output_shapes

:@g
dense_3/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/MaxMaxdense_3/Abs:y:0&dense_3/Max/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(R
dense_3/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @i
dense_3/mulMuldense_3/Max:output:0dense_3/mul/y:output:0*
T0*
_output_shapes

:X
dense_3/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `At
dense_3/truediv_1RealDivdense_3/mul:z:0dense_3/truediv_1/y:output:0*
T0*
_output_shapes

:R
dense_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3l
dense_3/addAddV2dense_3/truediv_1:z:0dense_3/add/y:output:0*
T0*
_output_shapes

:L
dense_3/LogLogdense_3/add:z:0*
T0*
_output_shapes

:X
dense_3/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?t
dense_3/truediv_2RealDivdense_3/Log:y:0dense_3/truediv_2/y:output:0*
T0*
_output_shapes

:V
dense_3/RoundRounddense_3/truediv_2:z:0*
T0*
_output_shapes

:T
dense_3/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @j
dense_3/Pow_1Powdense_3/Pow_1/x:output:0dense_3/Round:y:0*
T0*
_output_shapes

:R
dense_3/Abs_1Absdense_3/truediv:z:0*
T0*
_output_shapes

:@k
dense_3/truediv_3RealDivdense_3/Abs_1:y:0dense_3/Pow_1:z:0*
T0*
_output_shapes

:@T
dense_3/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?p
dense_3/add_1AddV2dense_3/truediv_3:z:0dense_3/add_1/y:output:0*
T0*
_output_shapes

:@R
dense_3/FloorFloordense_3/add_1:z:0*
T0*
_output_shapes

:@S
dense_3/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@i
dense_3/LessLessdense_3/Floor:y:0dense_3/Less/y:output:0*
T0*
_output_shapes

:@R
dense_3/SignSigndense_3/truediv:z:0*
T0*
_output_shapes

:@x
'dense_3/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      \
dense_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_3/ones_likeFill0dense_3/ones_like/Shape/shape_as_tensor:output:0 dense_3/ones_like/Const:output:0*
T0*
_output_shapes

:@T
dense_3/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `As
dense_3/mul_1Muldense_3/ones_like:output:0dense_3/mul_1/y:output:0*
T0*
_output_shapes

:@X
dense_3/truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @v
dense_3/truediv_4RealDivdense_3/mul_1:z:0dense_3/truediv_4/y:output:0*
T0*
_output_shapes

:@
dense_3/SelectV2SelectV2dense_3/Less:z:0dense_3/Floor:y:0dense_3/truediv_4:z:0*
T0*
_output_shapes

:@j
dense_3/mul_2Muldense_3/Sign:y:0dense_3/SelectV2:output:0*
T0*
_output_shapes

:@e
dense_3/Mul_3Muldense_3/truediv:z:0dense_3/mul_2:z:0*
T0*
_output_shapes

:@h
dense_3/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/MeanMeandense_3/Mul_3:z:0'dense_3/Mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(c
dense_3/Mul_4Muldense_3/mul_2:z:0dense_3/mul_2:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_1Meandense_3/Mul_4:z:0)dense_3/Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(T
dense_3/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_3/add_2AddV2dense_3/Mean_1:output:0dense_3/add_2/y:output:0*
T0*
_output_shapes

:o
dense_3/truediv_5RealDivdense_3/Mean:output:0dense_3/add_2:z:0*
T0*
_output_shapes

:T
dense_3/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3p
dense_3/add_3AddV2dense_3/truediv_5:z:0dense_3/add_3/y:output:0*
T0*
_output_shapes

:P
dense_3/Log_1Logdense_3/add_3:z:0*
T0*
_output_shapes

:X
dense_3/truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?v
dense_3/truediv_6RealDivdense_3/Log_1:y:0dense_3/truediv_6/y:output:0*
T0*
_output_shapes

:X
dense_3/Round_1Rounddense_3/truediv_6:z:0*
T0*
_output_shapes

:T
dense_3/Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_3/Pow_2Powdense_3/Pow_2/x:output:0dense_3/Round_1:y:0*
T0*
_output_shapes

:R
dense_3/Abs_2Absdense_3/truediv:z:0*
T0*
_output_shapes

:@k
dense_3/truediv_7RealDivdense_3/Abs_2:y:0dense_3/Pow_2:z:0*
T0*
_output_shapes

:@T
dense_3/add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?p
dense_3/add_4AddV2dense_3/truediv_7:z:0dense_3/add_4/y:output:0*
T0*
_output_shapes

:@T
dense_3/Floor_1Floordense_3/add_4:z:0*
T0*
_output_shapes

:@U
dense_3/Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_3/Less_1Lessdense_3/Floor_1:y:0dense_3/Less_1/y:output:0*
T0*
_output_shapes

:@T
dense_3/Sign_1Signdense_3/truediv:z:0*
T0*
_output_shapes

:@z
)dense_3/ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      ^
dense_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_3/ones_like_1Fill2dense_3/ones_like_1/Shape/shape_as_tensor:output:0"dense_3/ones_like_1/Const:output:0*
T0*
_output_shapes

:@T
dense_3/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Au
dense_3/mul_5Muldense_3/ones_like_1:output:0dense_3/mul_5/y:output:0*
T0*
_output_shapes

:@X
dense_3/truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @v
dense_3/truediv_8RealDivdense_3/mul_5:z:0dense_3/truediv_8/y:output:0*
T0*
_output_shapes

:@
dense_3/SelectV2_1SelectV2dense_3/Less_1:z:0dense_3/Floor_1:y:0dense_3/truediv_8:z:0*
T0*
_output_shapes

:@n
dense_3/mul_6Muldense_3/Sign_1:y:0dense_3/SelectV2_1:output:0*
T0*
_output_shapes

:@e
dense_3/Mul_7Muldense_3/truediv:z:0dense_3/mul_6:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_2Meandense_3/Mul_7:z:0)dense_3/Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(c
dense_3/Mul_8Muldense_3/mul_6:z:0dense_3/mul_6:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_3Meandense_3/Mul_8:z:0)dense_3/Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(T
dense_3/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_3/add_5AddV2dense_3/Mean_3:output:0dense_3/add_5/y:output:0*
T0*
_output_shapes

:q
dense_3/truediv_9RealDivdense_3/Mean_2:output:0dense_3/add_5:z:0*
T0*
_output_shapes

:T
dense_3/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3p
dense_3/add_6AddV2dense_3/truediv_9:z:0dense_3/add_6/y:output:0*
T0*
_output_shapes

:P
dense_3/Log_2Logdense_3/add_6:z:0*
T0*
_output_shapes

:Y
dense_3/truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_3/truediv_10RealDivdense_3/Log_2:y:0dense_3/truediv_10/y:output:0*
T0*
_output_shapes

:Y
dense_3/Round_2Rounddense_3/truediv_10:z:0*
T0*
_output_shapes

:T
dense_3/Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_3/Pow_3Powdense_3/Pow_3/x:output:0dense_3/Round_2:y:0*
T0*
_output_shapes

:R
dense_3/Abs_3Absdense_3/truediv:z:0*
T0*
_output_shapes

:@l
dense_3/truediv_11RealDivdense_3/Abs_3:y:0dense_3/Pow_3:z:0*
T0*
_output_shapes

:@T
dense_3/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?q
dense_3/add_7AddV2dense_3/truediv_11:z:0dense_3/add_7/y:output:0*
T0*
_output_shapes

:@T
dense_3/Floor_2Floordense_3/add_7:z:0*
T0*
_output_shapes

:@U
dense_3/Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_3/Less_2Lessdense_3/Floor_2:y:0dense_3/Less_2/y:output:0*
T0*
_output_shapes

:@T
dense_3/Sign_2Signdense_3/truediv:z:0*
T0*
_output_shapes

:@z
)dense_3/ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      ^
dense_3/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_3/ones_like_2Fill2dense_3/ones_like_2/Shape/shape_as_tensor:output:0"dense_3/ones_like_2/Const:output:0*
T0*
_output_shapes

:@T
dense_3/mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Au
dense_3/mul_9Muldense_3/ones_like_2:output:0dense_3/mul_9/y:output:0*
T0*
_output_shapes

:@Y
dense_3/truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @x
dense_3/truediv_12RealDivdense_3/mul_9:z:0dense_3/truediv_12/y:output:0*
T0*
_output_shapes

:@
dense_3/SelectV2_2SelectV2dense_3/Less_2:z:0dense_3/Floor_2:y:0dense_3/truediv_12:z:0*
T0*
_output_shapes

:@o
dense_3/mul_10Muldense_3/Sign_2:y:0dense_3/SelectV2_2:output:0*
T0*
_output_shapes

:@g
dense_3/Mul_11Muldense_3/truediv:z:0dense_3/mul_10:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_4Meandense_3/Mul_11:z:0)dense_3/Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(f
dense_3/Mul_12Muldense_3/mul_10:z:0dense_3/mul_10:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_5Meandense_3/Mul_12:z:0)dense_3/Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(T
dense_3/add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_3/add_8AddV2dense_3/Mean_5:output:0dense_3/add_8/y:output:0*
T0*
_output_shapes

:r
dense_3/truediv_13RealDivdense_3/Mean_4:output:0dense_3/add_8:z:0*
T0*
_output_shapes

:T
dense_3/add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3q
dense_3/add_9AddV2dense_3/truediv_13:z:0dense_3/add_9/y:output:0*
T0*
_output_shapes

:P
dense_3/Log_3Logdense_3/add_9:z:0*
T0*
_output_shapes

:Y
dense_3/truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_3/truediv_14RealDivdense_3/Log_3:y:0dense_3/truediv_14/y:output:0*
T0*
_output_shapes

:Y
dense_3/Round_3Rounddense_3/truediv_14:z:0*
T0*
_output_shapes

:T
dense_3/Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_3/Pow_4Powdense_3/Pow_4/x:output:0dense_3/Round_3:y:0*
T0*
_output_shapes

:R
dense_3/Abs_4Absdense_3/truediv:z:0*
T0*
_output_shapes

:@l
dense_3/truediv_15RealDivdense_3/Abs_4:y:0dense_3/Pow_4:z:0*
T0*
_output_shapes

:@U
dense_3/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?s
dense_3/add_10AddV2dense_3/truediv_15:z:0dense_3/add_10/y:output:0*
T0*
_output_shapes

:@U
dense_3/Floor_3Floordense_3/add_10:z:0*
T0*
_output_shapes

:@U
dense_3/Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_3/Less_3Lessdense_3/Floor_3:y:0dense_3/Less_3/y:output:0*
T0*
_output_shapes

:@T
dense_3/Sign_3Signdense_3/truediv:z:0*
T0*
_output_shapes

:@z
)dense_3/ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      ^
dense_3/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_3/ones_like_3Fill2dense_3/ones_like_3/Shape/shape_as_tensor:output:0"dense_3/ones_like_3/Const:output:0*
T0*
_output_shapes

:@U
dense_3/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Aw
dense_3/mul_13Muldense_3/ones_like_3:output:0dense_3/mul_13/y:output:0*
T0*
_output_shapes

:@Y
dense_3/truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @y
dense_3/truediv_16RealDivdense_3/mul_13:z:0dense_3/truediv_16/y:output:0*
T0*
_output_shapes

:@
dense_3/SelectV2_3SelectV2dense_3/Less_3:z:0dense_3/Floor_3:y:0dense_3/truediv_16:z:0*
T0*
_output_shapes

:@o
dense_3/mul_14Muldense_3/Sign_3:y:0dense_3/SelectV2_3:output:0*
T0*
_output_shapes

:@g
dense_3/Mul_15Muldense_3/truediv:z:0dense_3/mul_14:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_6Meandense_3/Mul_15:z:0)dense_3/Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(f
dense_3/Mul_16Muldense_3/mul_14:z:0dense_3/mul_14:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_7Meandense_3/Mul_16:z:0)dense_3/Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(U
dense_3/add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3t
dense_3/add_11AddV2dense_3/Mean_7:output:0dense_3/add_11/y:output:0*
T0*
_output_shapes

:s
dense_3/truediv_17RealDivdense_3/Mean_6:output:0dense_3/add_11:z:0*
T0*
_output_shapes

:U
dense_3/add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3s
dense_3/add_12AddV2dense_3/truediv_17:z:0dense_3/add_12/y:output:0*
T0*
_output_shapes

:Q
dense_3/Log_4Logdense_3/add_12:z:0*
T0*
_output_shapes

:Y
dense_3/truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_3/truediv_18RealDivdense_3/Log_4:y:0dense_3/truediv_18/y:output:0*
T0*
_output_shapes

:Y
dense_3/Round_4Rounddense_3/truediv_18:z:0*
T0*
_output_shapes

:T
dense_3/Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_3/Pow_5Powdense_3/Pow_5/x:output:0dense_3/Round_4:y:0*
T0*
_output_shapes

:R
dense_3/Abs_5Absdense_3/truediv:z:0*
T0*
_output_shapes

:@l
dense_3/truediv_19RealDivdense_3/Abs_5:y:0dense_3/Pow_5:z:0*
T0*
_output_shapes

:@U
dense_3/add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?s
dense_3/add_13AddV2dense_3/truediv_19:z:0dense_3/add_13/y:output:0*
T0*
_output_shapes

:@U
dense_3/Floor_4Floordense_3/add_13:z:0*
T0*
_output_shapes

:@U
dense_3/Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_3/Less_4Lessdense_3/Floor_4:y:0dense_3/Less_4/y:output:0*
T0*
_output_shapes

:@T
dense_3/Sign_4Signdense_3/truediv:z:0*
T0*
_output_shapes

:@z
)dense_3/ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      ^
dense_3/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_3/ones_like_4Fill2dense_3/ones_like_4/Shape/shape_as_tensor:output:0"dense_3/ones_like_4/Const:output:0*
T0*
_output_shapes

:@U
dense_3/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Aw
dense_3/mul_17Muldense_3/ones_like_4:output:0dense_3/mul_17/y:output:0*
T0*
_output_shapes

:@Y
dense_3/truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @y
dense_3/truediv_20RealDivdense_3/mul_17:z:0dense_3/truediv_20/y:output:0*
T0*
_output_shapes

:@
dense_3/SelectV2_4SelectV2dense_3/Less_4:z:0dense_3/Floor_4:y:0dense_3/truediv_20:z:0*
T0*
_output_shapes

:@o
dense_3/mul_18Muldense_3/Sign_4:y:0dense_3/SelectV2_4:output:0*
T0*
_output_shapes

:@g
dense_3/Mul_19Muldense_3/truediv:z:0dense_3/mul_18:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_8Meandense_3/Mul_19:z:0)dense_3/Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(f
dense_3/Mul_20Muldense_3/mul_18:z:0dense_3/mul_18:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_9Meandense_3/Mul_20:z:0)dense_3/Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(U
dense_3/add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3t
dense_3/add_14AddV2dense_3/Mean_9:output:0dense_3/add_14/y:output:0*
T0*
_output_shapes

:s
dense_3/truediv_21RealDivdense_3/Mean_8:output:0dense_3/add_14:z:0*
T0*
_output_shapes

:U
dense_3/add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3s
dense_3/add_15AddV2dense_3/truediv_21:z:0dense_3/add_15/y:output:0*
T0*
_output_shapes

:Q
dense_3/Log_5Logdense_3/add_15:z:0*
T0*
_output_shapes

:Y
dense_3/truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_3/truediv_22RealDivdense_3/Log_5:y:0dense_3/truediv_22/y:output:0*
T0*
_output_shapes

:Y
dense_3/Round_5Rounddense_3/truediv_22:z:0*
T0*
_output_shapes

:T
dense_3/Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_3/Pow_6Powdense_3/Pow_6/x:output:0dense_3/Round_5:y:0*
T0*
_output_shapes

:U
dense_3/mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Al
dense_3/mul_21Muldense_3/Pow_6:z:0dense_3/mul_21/y:output:0*
T0*
_output_shapes

:e
dense_3/mul_22Muldense_3/Cast:y:0dense_3/truediv:z:0*
T0*
_output_shapes

:@d
dense_3/mul_23Muldense_3/Cast:y:0dense_3/mul_18:z:0*
T0*
_output_shapes

:@Y
dense_3/truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ay
dense_3/truediv_23RealDivdense_3/mul_23:z:0dense_3/truediv_23/y:output:0*
T0*
_output_shapes

:@j
dense_3/mul_24Muldense_3/mul_21:z:0dense_3/truediv_23:z:0*
T0*
_output_shapes

:@O
dense_3/NegNegdense_3/mul_22:z:0*
T0*
_output_shapes

:@e
dense_3/add_16AddV2dense_3/Neg:y:0dense_3/mul_24:z:0*
T0*
_output_shapes

:@U
dense_3/mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
dense_3/mul_25Muldense_3/mul_25/x:output:0dense_3/add_16:z:0*
T0*
_output_shapes

:@a
dense_3/StopGradientStopGradientdense_3/mul_25:z:0*
T0*
_output_shapes

:@s
dense_3/add_17AddV2dense_3/mul_22:z:0dense_3/StopGradient:output:0*
T0*
_output_shapes

:@q
dense_3/MatMulMatMulre_lu_2/add_2:z:0dense_3/add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
dense_3/Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :Q
dense_3/Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : i
dense_3/Pow_7Powdense_3/Pow_7/x:output:0dense_3/Pow_7/y:output:0*
T0*
_output_shapes
: Y
dense_3/Cast_1Castdense_3/Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: v
dense_3/ReadVariableOp_1ReadVariableOp!dense_3_readvariableop_1_resource*
_output_shapes
:*
dtype0U
dense_3/mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aw
dense_3/mul_26Mul dense_3/ReadVariableOp_1:value:0dense_3/mul_26/y:output:0*
T0*
_output_shapes
:j
dense_3/truediv_24RealDivdense_3/mul_26:z:0dense_3/Cast_1:y:0*
T0*
_output_shapes
:Q
dense_3/Neg_1Negdense_3/truediv_24:z:0*
T0*
_output_shapes
:U
dense_3/Round_6Rounddense_3/truediv_24:z:0*
T0*
_output_shapes
:d
dense_3/add_18AddV2dense_3/Neg_1:y:0dense_3/Round_6:y:0*
T0*
_output_shapes
:_
dense_3/StopGradient_1StopGradientdense_3/add_18:z:0*
T0*
_output_shapes
:u
dense_3/add_19AddV2dense_3/truediv_24:z:0dense_3/StopGradient_1:output:0*
T0*
_output_shapes
:d
dense_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
dense_3/clip_by_value/MinimumMinimumdense_3/add_19:z:0(dense_3/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:\
dense_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á
dense_3/clip_by_valueMaximum!dense_3/clip_by_value/Minimum:z:0 dense_3/clip_by_value/y:output:0*
T0*
_output_shapes
:i
dense_3/mul_27Muldense_3/Cast_1:y:0dense_3/clip_by_value:z:0*
T0*
_output_shapes
:Y
dense_3/truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Au
dense_3/truediv_25RealDivdense_3/mul_27:z:0dense_3/truediv_25/y:output:0*
T0*
_output_shapes
:U
dense_3/mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
dense_3/mul_28Muldense_3/mul_28/x:output:0dense_3/truediv_25:z:0*
T0*
_output_shapes
:v
dense_3/ReadVariableOp_2ReadVariableOp!dense_3_readvariableop_1_resource*
_output_shapes
:*
dtype0[
dense_3/Neg_2Neg dense_3/ReadVariableOp_2:value:0*
T0*
_output_shapes
:c
dense_3/add_20AddV2dense_3/Neg_2:y:0dense_3/mul_28:z:0*
T0*
_output_shapes
:U
dense_3/mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
dense_3/mul_29Muldense_3/mul_29/x:output:0dense_3/add_20:z:0*
T0*
_output_shapes
:_
dense_3/StopGradient_2StopGradientdense_3/mul_29:z:0*
T0*
_output_shapes
:v
dense_3/ReadVariableOp_3ReadVariableOp!dense_3_readvariableop_1_resource*
_output_shapes
:*
dtype0
dense_3/add_21AddV2 dense_3/ReadVariableOp_3:value:0dense_3/StopGradient_2:output:0*
T0*
_output_shapes
:z
dense_3/BiasAddBiasAdddense_3/MatMul:product:0dense_3/add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ

NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense/ReadVariableOp^dense/ReadVariableOp_1^dense/ReadVariableOp_2^dense/ReadVariableOp_3^dense_1/ReadVariableOp^dense_1/ReadVariableOp_1^dense_1/ReadVariableOp_2^dense_1/ReadVariableOp_3^dense_2/ReadVariableOp^dense_2/ReadVariableOp_1^dense_2/ReadVariableOp_2^dense_2/ReadVariableOp_3^dense_3/ReadVariableOp^dense_3/ReadVariableOp_1^dense_3/ReadVariableOp_2^dense_3/ReadVariableOp_3*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿv: : : : : : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2,
dense/ReadVariableOpdense/ReadVariableOp20
dense/ReadVariableOp_1dense/ReadVariableOp_120
dense/ReadVariableOp_2dense/ReadVariableOp_220
dense/ReadVariableOp_3dense/ReadVariableOp_320
dense_1/ReadVariableOpdense_1/ReadVariableOp24
dense_1/ReadVariableOp_1dense_1/ReadVariableOp_124
dense_1/ReadVariableOp_2dense_1/ReadVariableOp_224
dense_1/ReadVariableOp_3dense_1/ReadVariableOp_320
dense_2/ReadVariableOpdense_2/ReadVariableOp24
dense_2/ReadVariableOp_1dense_2/ReadVariableOp_124
dense_2/ReadVariableOp_2dense_2/ReadVariableOp_224
dense_2/ReadVariableOp_3dense_2/ReadVariableOp_320
dense_3/ReadVariableOpdense_3/ReadVariableOp24
dense_3/ReadVariableOp_1dense_3/ReadVariableOp_124
dense_3/ReadVariableOp_2dense_3/ReadVariableOp_224
dense_3/ReadVariableOp_3dense_3/ReadVariableOp_3:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_nameinputs
æ¨
ý
@__inference_dense_layer_call_and_return_conditional_losses_75649

inputs)
readvariableop_resource:v@'
readvariableop_1_resource:@
identity¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:v@*
dtype0]
truedivRealDivReadVariableOp:value:0Cast:y:0*
T0*
_output_shapes

:v@@
AbsAbstruediv:z:0*
T0*
_output_shapes

:v@_
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: m
MaxMaxAbs:y:0Max/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Q
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes

:@P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A\
	truediv_1RealDivmul:z:0truediv_1/y:output:0*
T0*
_output_shapes

:@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3T
addAddV2truediv_1:z:0add/y:output:0*
T0*
_output_shapes

:@<
LogLogadd:z:0*
T0*
_output_shapes

:@P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?\
	truediv_2RealDivLog:y:0truediv_2/y:output:0*
T0*
_output_shapes

:@F
RoundRoundtruediv_2:z:0*
T0*
_output_shapes

:@L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @R
Pow_1PowPow_1/x:output:0	Round:y:0*
T0*
_output_shapes

:@B
Abs_1Abstruediv:z:0*
T0*
_output_shapes

:v@S
	truediv_3RealDiv	Abs_1:y:0	Pow_1:z:0*
T0*
_output_shapes

:v@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
add_1AddV2truediv_3:z:0add_1/y:output:0*
T0*
_output_shapes

:v@B
FloorFloor	add_1:z:0*
T0*
_output_shapes

:v@K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@Q
LessLess	Floor:y:0Less/y:output:0*
T0*
_output_shapes

:v@B
SignSigntruediv:z:0*
T0*
_output_shapes

:v@p
ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
	ones_likeFill(ones_like/Shape/shape_as_tensor:output:0ones_like/Const:output:0*
T0*
_output_shapes

:v@L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A[
mul_1Mulones_like:output:0mul_1/y:output:0*
T0*
_output_shapes

:v@P
truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
	truediv_4RealDiv	mul_1:z:0truediv_4/y:output:0*
T0*
_output_shapes

:v@a
SelectV2SelectV2Less:z:0	Floor:y:0truediv_4:z:0*
T0*
_output_shapes

:v@R
mul_2MulSign:y:0SelectV2:output:0*
T0*
_output_shapes

:v@M
Mul_3Multruediv:z:0	mul_2:z:0*
T0*
_output_shapes

:v@`
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: r
MeanMean	Mul_3:z:0Mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(K
Mul_4Mul	mul_2:z:0	mul_2:z:0*
T0*
_output_shapes

:v@b
Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_1Mean	Mul_4:z:0!Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_2AddV2Mean_1:output:0add_2/y:output:0*
T0*
_output_shapes

:@W
	truediv_5RealDivMean:output:0	add_2:z:0*
T0*
_output_shapes

:@L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3X
add_3AddV2truediv_5:z:0add_3/y:output:0*
T0*
_output_shapes

:@@
Log_1Log	add_3:z:0*
T0*
_output_shapes

:@P
truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?^
	truediv_6RealDiv	Log_1:y:0truediv_6/y:output:0*
T0*
_output_shapes

:@H
Round_1Roundtruediv_6:z:0*
T0*
_output_shapes

:@L
Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_2PowPow_2/x:output:0Round_1:y:0*
T0*
_output_shapes

:@B
Abs_2Abstruediv:z:0*
T0*
_output_shapes

:v@S
	truediv_7RealDiv	Abs_2:y:0	Pow_2:z:0*
T0*
_output_shapes

:v@L
add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
add_4AddV2truediv_7:z:0add_4/y:output:0*
T0*
_output_shapes

:v@D
Floor_1Floor	add_4:z:0*
T0*
_output_shapes

:v@M
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_1LessFloor_1:y:0Less_1/y:output:0*
T0*
_output_shapes

:v@D
Sign_1Signtruediv:z:0*
T0*
_output_shapes

:v@r
!ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_1Fill*ones_like_1/Shape/shape_as_tensor:output:0ones_like_1/Const:output:0*
T0*
_output_shapes

:v@L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A]
mul_5Mulones_like_1:output:0mul_5/y:output:0*
T0*
_output_shapes

:v@P
truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
	truediv_8RealDiv	mul_5:z:0truediv_8/y:output:0*
T0*
_output_shapes

:v@g

SelectV2_1SelectV2
Less_1:z:0Floor_1:y:0truediv_8:z:0*
T0*
_output_shapes

:v@V
mul_6Mul
Sign_1:y:0SelectV2_1:output:0*
T0*
_output_shapes

:v@M
Mul_7Multruediv:z:0	mul_6:z:0*
T0*
_output_shapes

:v@b
Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_2Mean	Mul_7:z:0!Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(K
Mul_8Mul	mul_6:z:0	mul_6:z:0*
T0*
_output_shapes

:v@b
Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_3Mean	Mul_8:z:0!Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_5AddV2Mean_3:output:0add_5/y:output:0*
T0*
_output_shapes

:@Y
	truediv_9RealDivMean_2:output:0	add_5:z:0*
T0*
_output_shapes

:@L
add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3X
add_6AddV2truediv_9:z:0add_6/y:output:0*
T0*
_output_shapes

:@@
Log_2Log	add_6:z:0*
T0*
_output_shapes

:@Q
truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_10RealDiv	Log_2:y:0truediv_10/y:output:0*
T0*
_output_shapes

:@I
Round_2Roundtruediv_10:z:0*
T0*
_output_shapes

:@L
Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_3PowPow_3/x:output:0Round_2:y:0*
T0*
_output_shapes

:@B
Abs_3Abstruediv:z:0*
T0*
_output_shapes

:v@T

truediv_11RealDiv	Abs_3:y:0	Pow_3:z:0*
T0*
_output_shapes

:v@L
add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Y
add_7AddV2truediv_11:z:0add_7/y:output:0*
T0*
_output_shapes

:v@D
Floor_2Floor	add_7:z:0*
T0*
_output_shapes

:v@M
Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_2LessFloor_2:y:0Less_2/y:output:0*
T0*
_output_shapes

:v@D
Sign_2Signtruediv:z:0*
T0*
_output_shapes

:v@r
!ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_2Fill*ones_like_2/Shape/shape_as_tensor:output:0ones_like_2/Const:output:0*
T0*
_output_shapes

:v@L
mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A]
mul_9Mulones_like_2:output:0mul_9/y:output:0*
T0*
_output_shapes

:v@Q
truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @`

truediv_12RealDiv	mul_9:z:0truediv_12/y:output:0*
T0*
_output_shapes

:v@h

SelectV2_2SelectV2
Less_2:z:0Floor_2:y:0truediv_12:z:0*
T0*
_output_shapes

:v@W
mul_10Mul
Sign_2:y:0SelectV2_2:output:0*
T0*
_output_shapes

:v@O
Mul_11Multruediv:z:0
mul_10:z:0*
T0*
_output_shapes

:v@b
Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_4Mean
Mul_11:z:0!Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_12Mul
mul_10:z:0
mul_10:z:0*
T0*
_output_shapes

:v@b
Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_5Mean
Mul_12:z:0!Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_8AddV2Mean_5:output:0add_8/y:output:0*
T0*
_output_shapes

:@Z

truediv_13RealDivMean_4:output:0	add_8:z:0*
T0*
_output_shapes

:@L
add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Y
add_9AddV2truediv_13:z:0add_9/y:output:0*
T0*
_output_shapes

:@@
Log_3Log	add_9:z:0*
T0*
_output_shapes

:@Q
truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_14RealDiv	Log_3:y:0truediv_14/y:output:0*
T0*
_output_shapes

:@I
Round_3Roundtruediv_14:z:0*
T0*
_output_shapes

:@L
Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_4PowPow_4/x:output:0Round_3:y:0*
T0*
_output_shapes

:@B
Abs_4Abstruediv:z:0*
T0*
_output_shapes

:v@T

truediv_15RealDiv	Abs_4:y:0	Pow_4:z:0*
T0*
_output_shapes

:v@M
add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_10AddV2truediv_15:z:0add_10/y:output:0*
T0*
_output_shapes

:v@E
Floor_3Floor
add_10:z:0*
T0*
_output_shapes

:v@M
Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_3LessFloor_3:y:0Less_3/y:output:0*
T0*
_output_shapes

:v@D
Sign_3Signtruediv:z:0*
T0*
_output_shapes

:v@r
!ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   V
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_3Fill*ones_like_3/Shape/shape_as_tensor:output:0ones_like_3/Const:output:0*
T0*
_output_shapes

:v@M
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A_
mul_13Mulones_like_3:output:0mul_13/y:output:0*
T0*
_output_shapes

:v@Q
truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @a

truediv_16RealDiv
mul_13:z:0truediv_16/y:output:0*
T0*
_output_shapes

:v@h

SelectV2_3SelectV2
Less_3:z:0Floor_3:y:0truediv_16:z:0*
T0*
_output_shapes

:v@W
mul_14Mul
Sign_3:y:0SelectV2_3:output:0*
T0*
_output_shapes

:v@O
Mul_15Multruediv:z:0
mul_14:z:0*
T0*
_output_shapes

:v@b
Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_6Mean
Mul_15:z:0!Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_16Mul
mul_14:z:0
mul_14:z:0*
T0*
_output_shapes

:v@b
Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_7Mean
Mul_16:z:0!Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(M
add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3\
add_11AddV2Mean_7:output:0add_11/y:output:0*
T0*
_output_shapes

:@[

truediv_17RealDivMean_6:output:0
add_11:z:0*
T0*
_output_shapes

:@M
add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3[
add_12AddV2truediv_17:z:0add_12/y:output:0*
T0*
_output_shapes

:@A
Log_4Log
add_12:z:0*
T0*
_output_shapes

:@Q
truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_18RealDiv	Log_4:y:0truediv_18/y:output:0*
T0*
_output_shapes

:@I
Round_4Roundtruediv_18:z:0*
T0*
_output_shapes

:@L
Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_5PowPow_5/x:output:0Round_4:y:0*
T0*
_output_shapes

:@B
Abs_5Abstruediv:z:0*
T0*
_output_shapes

:v@T

truediv_19RealDiv	Abs_5:y:0	Pow_5:z:0*
T0*
_output_shapes

:v@M
add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_13AddV2truediv_19:z:0add_13/y:output:0*
T0*
_output_shapes

:v@E
Floor_4Floor
add_13:z:0*
T0*
_output_shapes

:v@M
Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_4LessFloor_4:y:0Less_4/y:output:0*
T0*
_output_shapes

:v@D
Sign_4Signtruediv:z:0*
T0*
_output_shapes

:v@r
!ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   V
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_4Fill*ones_like_4/Shape/shape_as_tensor:output:0ones_like_4/Const:output:0*
T0*
_output_shapes

:v@M
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A_
mul_17Mulones_like_4:output:0mul_17/y:output:0*
T0*
_output_shapes

:v@Q
truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @a

truediv_20RealDiv
mul_17:z:0truediv_20/y:output:0*
T0*
_output_shapes

:v@h

SelectV2_4SelectV2
Less_4:z:0Floor_4:y:0truediv_20:z:0*
T0*
_output_shapes

:v@W
mul_18Mul
Sign_4:y:0SelectV2_4:output:0*
T0*
_output_shapes

:v@O
Mul_19Multruediv:z:0
mul_18:z:0*
T0*
_output_shapes

:v@b
Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_8Mean
Mul_19:z:0!Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_20Mul
mul_18:z:0
mul_18:z:0*
T0*
_output_shapes

:v@b
Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_9Mean
Mul_20:z:0!Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(M
add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3\
add_14AddV2Mean_9:output:0add_14/y:output:0*
T0*
_output_shapes

:@[

truediv_21RealDivMean_8:output:0
add_14:z:0*
T0*
_output_shapes

:@M
add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3[
add_15AddV2truediv_21:z:0add_15/y:output:0*
T0*
_output_shapes

:@A
Log_5Log
add_15:z:0*
T0*
_output_shapes

:@Q
truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_22RealDiv	Log_5:y:0truediv_22/y:output:0*
T0*
_output_shapes

:@I
Round_5Roundtruediv_22:z:0*
T0*
_output_shapes

:@L
Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_6PowPow_6/x:output:0Round_5:y:0*
T0*
_output_shapes

:@M
mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AT
mul_21Mul	Pow_6:z:0mul_21/y:output:0*
T0*
_output_shapes

:@M
mul_22MulCast:y:0truediv:z:0*
T0*
_output_shapes

:v@L
mul_23MulCast:y:0
mul_18:z:0*
T0*
_output_shapes

:v@Q
truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aa

truediv_23RealDiv
mul_23:z:0truediv_23/y:output:0*
T0*
_output_shapes

:v@R
mul_24Mul
mul_21:z:0truediv_23:z:0*
T0*
_output_shapes

:v@?
NegNeg
mul_22:z:0*
T0*
_output_shapes

:v@M
add_16AddV2Neg:y:0
mul_24:z:0*
T0*
_output_shapes

:v@M
mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mul_25Mulmul_25/x:output:0
add_16:z:0*
T0*
_output_shapes

:v@Q
StopGradientStopGradient
mul_25:z:0*
T0*
_output_shapes

:v@[
add_17AddV2
mul_22:z:0StopGradient:output:0*
T0*
_output_shapes

:v@V
MatMulMatMulinputs
add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@I
Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_7PowPow_7/x:output:0Pow_7/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0M
mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A_
mul_26MulReadVariableOp_1:value:0mul_26/y:output:0*
T0*
_output_shapes
:@R

truediv_24RealDiv
mul_26:z:0
Cast_1:y:0*
T0*
_output_shapes
:@A
Neg_1Negtruediv_24:z:0*
T0*
_output_shapes
:@E
Round_6Roundtruediv_24:z:0*
T0*
_output_shapes
:@L
add_18AddV2	Neg_1:y:0Round_6:y:0*
T0*
_output_shapes
:@O
StopGradient_1StopGradient
add_18:z:0*
T0*
_output_shapes
:@]
add_19AddV2truediv_24:z:0StopGradient_1:output:0*
T0*
_output_shapes
:@\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@s
clip_by_value/MinimumMinimum
add_19:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:@T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ár
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:@Q
mul_27Mul
Cast_1:y:0clip_by_value:z:0*
T0*
_output_shapes
:@Q
truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A]

truediv_25RealDiv
mul_27:z:0truediv_25/y:output:0*
T0*
_output_shapes
:@M
mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mul_28Mulmul_28/x:output:0truediv_25:z:0*
T0*
_output_shapes
:@f
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0K
Neg_2NegReadVariableOp_2:value:0*
T0*
_output_shapes
:@K
add_20AddV2	Neg_2:y:0
mul_28:z:0*
T0*
_output_shapes
:@M
mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Q
mul_29Mulmul_29/x:output:0
add_20:z:0*
T0*
_output_shapes
:@O
StopGradient_2StopGradient
mul_29:z:0*
T0*
_output_shapes
:@f
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0g
add_21AddV2ReadVariableOp_3:value:0StopGradient_2:output:0*
T0*
_output_shapes
:@b
BiasAddBiasAddMatMul:product:0
add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿv: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_3:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_nameinputs
ò	
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_80007

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
%
é
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_75389

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

E
)__inference_dropout_1_layer_call_fn_79985

inputs
identity¯
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_75951`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
º

%__inference_dense_layer_call_fn_79279

inputs
unknown:v@
	unknown_0:@
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75649o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿv: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_nameinputs
²
ñ
-__inference_my_sequential_layer_call_fn_77121

inputs
unknown:v@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_my_sequential_layer_call_and_return_conditional_losses_76776o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿv: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_nameinputs
¤
Ð
5__inference_batch_normalization_2_layer_call_fn_80305

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_75389o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
êü
Ó
H__inference_my_sequential_layer_call_and_return_conditional_losses_78164

inputs/
dense_readvariableop_resource:v@-
dense_readvariableop_1_resource:@C
5batch_normalization_batchnorm_readvariableop_resource:@G
9batch_normalization_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_batchnorm_readvariableop_1_resource:@E
7batch_normalization_batchnorm_readvariableop_2_resource:@1
dense_1_readvariableop_resource:@@/
!dense_1_readvariableop_1_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_1_batchnorm_readvariableop_1_resource:@G
9batch_normalization_1_batchnorm_readvariableop_2_resource:@1
dense_2_readvariableop_resource:@@/
!dense_2_readvariableop_1_resource:@E
7batch_normalization_2_batchnorm_readvariableop_resource:@I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_2_batchnorm_readvariableop_1_resource:@G
9batch_normalization_2_batchnorm_readvariableop_2_resource:@1
dense_3_readvariableop_resource:@/
!dense_3_readvariableop_1_resource:
identity¢,batch_normalization/batchnorm/ReadVariableOp¢.batch_normalization/batchnorm/ReadVariableOp_1¢.batch_normalization/batchnorm/ReadVariableOp_2¢0batch_normalization/batchnorm/mul/ReadVariableOp¢.batch_normalization_1/batchnorm/ReadVariableOp¢0batch_normalization_1/batchnorm/ReadVariableOp_1¢0batch_normalization_1/batchnorm/ReadVariableOp_2¢2batch_normalization_1/batchnorm/mul/ReadVariableOp¢.batch_normalization_2/batchnorm/ReadVariableOp¢0batch_normalization_2/batchnorm/ReadVariableOp_1¢0batch_normalization_2/batchnorm/ReadVariableOp_2¢2batch_normalization_2/batchnorm/mul/ReadVariableOp¢dense/ReadVariableOp¢dense/ReadVariableOp_1¢dense/ReadVariableOp_2¢dense/ReadVariableOp_3¢dense_1/ReadVariableOp¢dense_1/ReadVariableOp_1¢dense_1/ReadVariableOp_2¢dense_1/ReadVariableOp_3¢dense_2/ReadVariableOp¢dense_2/ReadVariableOp_1¢dense_2/ReadVariableOp_2¢dense_2/ReadVariableOp_3¢dense_3/ReadVariableOp¢dense_3/ReadVariableOp_1¢dense_3/ReadVariableOp_2¢dense_3/ReadVariableOp_3M
dense/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :M
dense/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : ]
	dense/PowPowdense/Pow/x:output:0dense/Pow/y:output:0*
T0*
_output_shapes
: Q

dense/CastCastdense/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: r
dense/ReadVariableOpReadVariableOpdense_readvariableop_resource*
_output_shapes

:v@*
dtype0o
dense/truedivRealDivdense/ReadVariableOp:value:0dense/Cast:y:0*
T0*
_output_shapes

:v@L
	dense/AbsAbsdense/truediv:z:0*
T0*
_output_shapes

:v@e
dense/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
	dense/MaxMaxdense/Abs:y:0$dense/Max/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(P
dense/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @c
	dense/mulMuldense/Max:output:0dense/mul/y:output:0*
T0*
_output_shapes

:@V
dense/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `An
dense/truediv_1RealDivdense/mul:z:0dense/truediv_1/y:output:0*
T0*
_output_shapes

:@P
dense/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3f
	dense/addAddV2dense/truediv_1:z:0dense/add/y:output:0*
T0*
_output_shapes

:@H
	dense/LogLogdense/add:z:0*
T0*
_output_shapes

:@V
dense/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?n
dense/truediv_2RealDivdense/Log:y:0dense/truediv_2/y:output:0*
T0*
_output_shapes

:@R
dense/RoundRounddense/truediv_2:z:0*
T0*
_output_shapes

:@R
dense/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dense/Pow_1Powdense/Pow_1/x:output:0dense/Round:y:0*
T0*
_output_shapes

:@N
dense/Abs_1Absdense/truediv:z:0*
T0*
_output_shapes

:v@e
dense/truediv_3RealDivdense/Abs_1:y:0dense/Pow_1:z:0*
T0*
_output_shapes

:v@R
dense/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?j
dense/add_1AddV2dense/truediv_3:z:0dense/add_1/y:output:0*
T0*
_output_shapes

:v@N
dense/FloorFloordense/add_1:z:0*
T0*
_output_shapes

:v@Q
dense/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@c

dense/LessLessdense/Floor:y:0dense/Less/y:output:0*
T0*
_output_shapes

:v@N

dense/SignSigndense/truediv:z:0*
T0*
_output_shapes

:v@v
%dense/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   Z
dense/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense/ones_likeFill.dense/ones_like/Shape/shape_as_tensor:output:0dense/ones_like/Const:output:0*
T0*
_output_shapes

:v@R
dense/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Am
dense/mul_1Muldense/ones_like:output:0dense/mul_1/y:output:0*
T0*
_output_shapes

:v@V
dense/truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @p
dense/truediv_4RealDivdense/mul_1:z:0dense/truediv_4/y:output:0*
T0*
_output_shapes

:v@y
dense/SelectV2SelectV2dense/Less:z:0dense/Floor:y:0dense/truediv_4:z:0*
T0*
_output_shapes

:v@d
dense/mul_2Muldense/Sign:y:0dense/SelectV2:output:0*
T0*
_output_shapes

:v@_
dense/Mul_3Muldense/truediv:z:0dense/mul_2:z:0*
T0*
_output_shapes

:v@f
dense/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 

dense/MeanMeandense/Mul_3:z:0%dense/Mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(]
dense/Mul_4Muldense/mul_2:z:0dense/mul_2:z:0*
T0*
_output_shapes

:v@h
dense/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_1Meandense/Mul_4:z:0'dense/Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(R
dense/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3l
dense/add_2AddV2dense/Mean_1:output:0dense/add_2/y:output:0*
T0*
_output_shapes

:@i
dense/truediv_5RealDivdense/Mean:output:0dense/add_2:z:0*
T0*
_output_shapes

:@R
dense/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3j
dense/add_3AddV2dense/truediv_5:z:0dense/add_3/y:output:0*
T0*
_output_shapes

:@L
dense/Log_1Logdense/add_3:z:0*
T0*
_output_shapes

:@V
dense/truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?p
dense/truediv_6RealDivdense/Log_1:y:0dense/truediv_6/y:output:0*
T0*
_output_shapes

:@T
dense/Round_1Rounddense/truediv_6:z:0*
T0*
_output_shapes

:@R
dense/Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dense/Pow_2Powdense/Pow_2/x:output:0dense/Round_1:y:0*
T0*
_output_shapes

:@N
dense/Abs_2Absdense/truediv:z:0*
T0*
_output_shapes

:v@e
dense/truediv_7RealDivdense/Abs_2:y:0dense/Pow_2:z:0*
T0*
_output_shapes

:v@R
dense/add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?j
dense/add_4AddV2dense/truediv_7:z:0dense/add_4/y:output:0*
T0*
_output_shapes

:v@P
dense/Floor_1Floordense/add_4:z:0*
T0*
_output_shapes

:v@S
dense/Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@i
dense/Less_1Lessdense/Floor_1:y:0dense/Less_1/y:output:0*
T0*
_output_shapes

:v@P
dense/Sign_1Signdense/truediv:z:0*
T0*
_output_shapes

:v@x
'dense/ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   \
dense/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense/ones_like_1Fill0dense/ones_like_1/Shape/shape_as_tensor:output:0 dense/ones_like_1/Const:output:0*
T0*
_output_shapes

:v@R
dense/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Ao
dense/mul_5Muldense/ones_like_1:output:0dense/mul_5/y:output:0*
T0*
_output_shapes

:v@V
dense/truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @p
dense/truediv_8RealDivdense/mul_5:z:0dense/truediv_8/y:output:0*
T0*
_output_shapes

:v@
dense/SelectV2_1SelectV2dense/Less_1:z:0dense/Floor_1:y:0dense/truediv_8:z:0*
T0*
_output_shapes

:v@h
dense/mul_6Muldense/Sign_1:y:0dense/SelectV2_1:output:0*
T0*
_output_shapes

:v@_
dense/Mul_7Muldense/truediv:z:0dense/mul_6:z:0*
T0*
_output_shapes

:v@h
dense/Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_2Meandense/Mul_7:z:0'dense/Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(]
dense/Mul_8Muldense/mul_6:z:0dense/mul_6:z:0*
T0*
_output_shapes

:v@h
dense/Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_3Meandense/Mul_8:z:0'dense/Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(R
dense/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3l
dense/add_5AddV2dense/Mean_3:output:0dense/add_5/y:output:0*
T0*
_output_shapes

:@k
dense/truediv_9RealDivdense/Mean_2:output:0dense/add_5:z:0*
T0*
_output_shapes

:@R
dense/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3j
dense/add_6AddV2dense/truediv_9:z:0dense/add_6/y:output:0*
T0*
_output_shapes

:@L
dense/Log_2Logdense/add_6:z:0*
T0*
_output_shapes

:@W
dense/truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?r
dense/truediv_10RealDivdense/Log_2:y:0dense/truediv_10/y:output:0*
T0*
_output_shapes

:@U
dense/Round_2Rounddense/truediv_10:z:0*
T0*
_output_shapes

:@R
dense/Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dense/Pow_3Powdense/Pow_3/x:output:0dense/Round_2:y:0*
T0*
_output_shapes

:@N
dense/Abs_3Absdense/truediv:z:0*
T0*
_output_shapes

:v@f
dense/truediv_11RealDivdense/Abs_3:y:0dense/Pow_3:z:0*
T0*
_output_shapes

:v@R
dense/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?k
dense/add_7AddV2dense/truediv_11:z:0dense/add_7/y:output:0*
T0*
_output_shapes

:v@P
dense/Floor_2Floordense/add_7:z:0*
T0*
_output_shapes

:v@S
dense/Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@i
dense/Less_2Lessdense/Floor_2:y:0dense/Less_2/y:output:0*
T0*
_output_shapes

:v@P
dense/Sign_2Signdense/truediv:z:0*
T0*
_output_shapes

:v@x
'dense/ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   \
dense/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense/ones_like_2Fill0dense/ones_like_2/Shape/shape_as_tensor:output:0 dense/ones_like_2/Const:output:0*
T0*
_output_shapes

:v@R
dense/mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Ao
dense/mul_9Muldense/ones_like_2:output:0dense/mul_9/y:output:0*
T0*
_output_shapes

:v@W
dense/truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @r
dense/truediv_12RealDivdense/mul_9:z:0dense/truediv_12/y:output:0*
T0*
_output_shapes

:v@
dense/SelectV2_2SelectV2dense/Less_2:z:0dense/Floor_2:y:0dense/truediv_12:z:0*
T0*
_output_shapes

:v@i
dense/mul_10Muldense/Sign_2:y:0dense/SelectV2_2:output:0*
T0*
_output_shapes

:v@a
dense/Mul_11Muldense/truediv:z:0dense/mul_10:z:0*
T0*
_output_shapes

:v@h
dense/Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_4Meandense/Mul_11:z:0'dense/Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(`
dense/Mul_12Muldense/mul_10:z:0dense/mul_10:z:0*
T0*
_output_shapes

:v@h
dense/Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_5Meandense/Mul_12:z:0'dense/Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(R
dense/add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3l
dense/add_8AddV2dense/Mean_5:output:0dense/add_8/y:output:0*
T0*
_output_shapes

:@l
dense/truediv_13RealDivdense/Mean_4:output:0dense/add_8:z:0*
T0*
_output_shapes

:@R
dense/add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3k
dense/add_9AddV2dense/truediv_13:z:0dense/add_9/y:output:0*
T0*
_output_shapes

:@L
dense/Log_3Logdense/add_9:z:0*
T0*
_output_shapes

:@W
dense/truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?r
dense/truediv_14RealDivdense/Log_3:y:0dense/truediv_14/y:output:0*
T0*
_output_shapes

:@U
dense/Round_3Rounddense/truediv_14:z:0*
T0*
_output_shapes

:@R
dense/Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dense/Pow_4Powdense/Pow_4/x:output:0dense/Round_3:y:0*
T0*
_output_shapes

:@N
dense/Abs_4Absdense/truediv:z:0*
T0*
_output_shapes

:v@f
dense/truediv_15RealDivdense/Abs_4:y:0dense/Pow_4:z:0*
T0*
_output_shapes

:v@S
dense/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?m
dense/add_10AddV2dense/truediv_15:z:0dense/add_10/y:output:0*
T0*
_output_shapes

:v@Q
dense/Floor_3Floordense/add_10:z:0*
T0*
_output_shapes

:v@S
dense/Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@i
dense/Less_3Lessdense/Floor_3:y:0dense/Less_3/y:output:0*
T0*
_output_shapes

:v@P
dense/Sign_3Signdense/truediv:z:0*
T0*
_output_shapes

:v@x
'dense/ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   \
dense/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense/ones_like_3Fill0dense/ones_like_3/Shape/shape_as_tensor:output:0 dense/ones_like_3/Const:output:0*
T0*
_output_shapes

:v@S
dense/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Aq
dense/mul_13Muldense/ones_like_3:output:0dense/mul_13/y:output:0*
T0*
_output_shapes

:v@W
dense/truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @s
dense/truediv_16RealDivdense/mul_13:z:0dense/truediv_16/y:output:0*
T0*
_output_shapes

:v@
dense/SelectV2_3SelectV2dense/Less_3:z:0dense/Floor_3:y:0dense/truediv_16:z:0*
T0*
_output_shapes

:v@i
dense/mul_14Muldense/Sign_3:y:0dense/SelectV2_3:output:0*
T0*
_output_shapes

:v@a
dense/Mul_15Muldense/truediv:z:0dense/mul_14:z:0*
T0*
_output_shapes

:v@h
dense/Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_6Meandense/Mul_15:z:0'dense/Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(`
dense/Mul_16Muldense/mul_14:z:0dense/mul_14:z:0*
T0*
_output_shapes

:v@h
dense/Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_7Meandense/Mul_16:z:0'dense/Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(S
dense/add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3n
dense/add_11AddV2dense/Mean_7:output:0dense/add_11/y:output:0*
T0*
_output_shapes

:@m
dense/truediv_17RealDivdense/Mean_6:output:0dense/add_11:z:0*
T0*
_output_shapes

:@S
dense/add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3m
dense/add_12AddV2dense/truediv_17:z:0dense/add_12/y:output:0*
T0*
_output_shapes

:@M
dense/Log_4Logdense/add_12:z:0*
T0*
_output_shapes

:@W
dense/truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?r
dense/truediv_18RealDivdense/Log_4:y:0dense/truediv_18/y:output:0*
T0*
_output_shapes

:@U
dense/Round_4Rounddense/truediv_18:z:0*
T0*
_output_shapes

:@R
dense/Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dense/Pow_5Powdense/Pow_5/x:output:0dense/Round_4:y:0*
T0*
_output_shapes

:@N
dense/Abs_5Absdense/truediv:z:0*
T0*
_output_shapes

:v@f
dense/truediv_19RealDivdense/Abs_5:y:0dense/Pow_5:z:0*
T0*
_output_shapes

:v@S
dense/add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?m
dense/add_13AddV2dense/truediv_19:z:0dense/add_13/y:output:0*
T0*
_output_shapes

:v@Q
dense/Floor_4Floordense/add_13:z:0*
T0*
_output_shapes

:v@S
dense/Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@i
dense/Less_4Lessdense/Floor_4:y:0dense/Less_4/y:output:0*
T0*
_output_shapes

:v@P
dense/Sign_4Signdense/truediv:z:0*
T0*
_output_shapes

:v@x
'dense/ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   \
dense/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense/ones_like_4Fill0dense/ones_like_4/Shape/shape_as_tensor:output:0 dense/ones_like_4/Const:output:0*
T0*
_output_shapes

:v@S
dense/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Aq
dense/mul_17Muldense/ones_like_4:output:0dense/mul_17/y:output:0*
T0*
_output_shapes

:v@W
dense/truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @s
dense/truediv_20RealDivdense/mul_17:z:0dense/truediv_20/y:output:0*
T0*
_output_shapes

:v@
dense/SelectV2_4SelectV2dense/Less_4:z:0dense/Floor_4:y:0dense/truediv_20:z:0*
T0*
_output_shapes

:v@i
dense/mul_18Muldense/Sign_4:y:0dense/SelectV2_4:output:0*
T0*
_output_shapes

:v@a
dense/Mul_19Muldense/truediv:z:0dense/mul_18:z:0*
T0*
_output_shapes

:v@h
dense/Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_8Meandense/Mul_19:z:0'dense/Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(`
dense/Mul_20Muldense/mul_18:z:0dense/mul_18:z:0*
T0*
_output_shapes

:v@h
dense/Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Mean_9Meandense/Mul_20:z:0'dense/Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(S
dense/add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3n
dense/add_14AddV2dense/Mean_9:output:0dense/add_14/y:output:0*
T0*
_output_shapes

:@m
dense/truediv_21RealDivdense/Mean_8:output:0dense/add_14:z:0*
T0*
_output_shapes

:@S
dense/add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3m
dense/add_15AddV2dense/truediv_21:z:0dense/add_15/y:output:0*
T0*
_output_shapes

:@M
dense/Log_5Logdense/add_15:z:0*
T0*
_output_shapes

:@W
dense/truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?r
dense/truediv_22RealDivdense/Log_5:y:0dense/truediv_22/y:output:0*
T0*
_output_shapes

:@U
dense/Round_5Rounddense/truediv_22:z:0*
T0*
_output_shapes

:@R
dense/Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dense/Pow_6Powdense/Pow_6/x:output:0dense/Round_5:y:0*
T0*
_output_shapes

:@S
dense/mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Af
dense/mul_21Muldense/Pow_6:z:0dense/mul_21/y:output:0*
T0*
_output_shapes

:@_
dense/mul_22Muldense/Cast:y:0dense/truediv:z:0*
T0*
_output_shapes

:v@^
dense/mul_23Muldense/Cast:y:0dense/mul_18:z:0*
T0*
_output_shapes

:v@W
dense/truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   As
dense/truediv_23RealDivdense/mul_23:z:0dense/truediv_23/y:output:0*
T0*
_output_shapes

:v@d
dense/mul_24Muldense/mul_21:z:0dense/truediv_23:z:0*
T0*
_output_shapes

:v@K
	dense/NegNegdense/mul_22:z:0*
T0*
_output_shapes

:v@_
dense/add_16AddV2dense/Neg:y:0dense/mul_24:z:0*
T0*
_output_shapes

:v@S
dense/mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
dense/mul_25Muldense/mul_25/x:output:0dense/add_16:z:0*
T0*
_output_shapes

:v@]
dense/StopGradientStopGradientdense/mul_25:z:0*
T0*
_output_shapes

:v@m
dense/add_17AddV2dense/mul_22:z:0dense/StopGradient:output:0*
T0*
_output_shapes

:v@b
dense/MatMulMatMulinputsdense/add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@O
dense/Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :O
dense/Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : c
dense/Pow_7Powdense/Pow_7/x:output:0dense/Pow_7/y:output:0*
T0*
_output_shapes
: U
dense/Cast_1Castdense/Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: r
dense/ReadVariableOp_1ReadVariableOpdense_readvariableop_1_resource*
_output_shapes
:@*
dtype0S
dense/mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aq
dense/mul_26Muldense/ReadVariableOp_1:value:0dense/mul_26/y:output:0*
T0*
_output_shapes
:@d
dense/truediv_24RealDivdense/mul_26:z:0dense/Cast_1:y:0*
T0*
_output_shapes
:@M
dense/Neg_1Negdense/truediv_24:z:0*
T0*
_output_shapes
:@Q
dense/Round_6Rounddense/truediv_24:z:0*
T0*
_output_shapes
:@^
dense/add_18AddV2dense/Neg_1:y:0dense/Round_6:y:0*
T0*
_output_shapes
:@[
dense/StopGradient_1StopGradientdense/add_18:z:0*
T0*
_output_shapes
:@o
dense/add_19AddV2dense/truediv_24:z:0dense/StopGradient_1:output:0*
T0*
_output_shapes
:@b
dense/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
dense/clip_by_value/MinimumMinimumdense/add_19:z:0&dense/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:@Z
dense/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á
dense/clip_by_valueMaximumdense/clip_by_value/Minimum:z:0dense/clip_by_value/y:output:0*
T0*
_output_shapes
:@c
dense/mul_27Muldense/Cast_1:y:0dense/clip_by_value:z:0*
T0*
_output_shapes
:@W
dense/truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ao
dense/truediv_25RealDivdense/mul_27:z:0dense/truediv_25/y:output:0*
T0*
_output_shapes
:@S
dense/mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
dense/mul_28Muldense/mul_28/x:output:0dense/truediv_25:z:0*
T0*
_output_shapes
:@r
dense/ReadVariableOp_2ReadVariableOpdense_readvariableop_1_resource*
_output_shapes
:@*
dtype0W
dense/Neg_2Negdense/ReadVariableOp_2:value:0*
T0*
_output_shapes
:@]
dense/add_20AddV2dense/Neg_2:y:0dense/mul_28:z:0*
T0*
_output_shapes
:@S
dense/mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?c
dense/mul_29Muldense/mul_29/x:output:0dense/add_20:z:0*
T0*
_output_shapes
:@[
dense/StopGradient_2StopGradientdense/mul_29:z:0*
T0*
_output_shapes
:@r
dense/ReadVariableOp_3ReadVariableOpdense_readvariableop_1_resource*
_output_shapes
:@*
dtype0y
dense/add_21AddV2dense/ReadVariableOp_3:value:0dense/StopGradient_2:output:0*
T0*
_output_shapes
:@t
dense/BiasAddBiasAdddense/MatMul:product:0dense/add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:³
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@¦
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0°
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@
#batch_normalization/batchnorm/mul_1Muldense/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0®
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@¢
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0®
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@®
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
dropout/IdentityIdentity'batch_normalization/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_

re_lu/SignSigndropout/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
	re_lu/AbsAbsre_lu/Sign:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
re_lu/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
	re_lu/subSubre_lu/sub/x:output:0re_lu/Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
	re_lu/addAddV2re_lu/Sign:y:0re_lu/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_

re_lu/TanhTanhdropout/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
	re_lu/NegNegre_lu/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
re_lu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
	re_lu/mulMulre_lu/mul/x:output:0re_lu/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
re_lu/add_1AddV2re_lu/Neg:y:0re_lu/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
re_lu/StopGradientStopGradientre_lu/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
re_lu/add_2AddV2re_lu/Tanh:y:0re_lu/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@O
dense_1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :O
dense_1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : c
dense_1/PowPowdense_1/Pow/x:output:0dense_1/Pow/y:output:0*
T0*
_output_shapes
: U
dense_1/CastCastdense_1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: v
dense_1/ReadVariableOpReadVariableOpdense_1_readvariableop_resource*
_output_shapes

:@@*
dtype0u
dense_1/truedivRealDivdense_1/ReadVariableOp:value:0dense_1/Cast:y:0*
T0*
_output_shapes

:@@P
dense_1/AbsAbsdense_1/truediv:z:0*
T0*
_output_shapes

:@@g
dense_1/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/MaxMaxdense_1/Abs:y:0&dense_1/Max/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(R
dense_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @i
dense_1/mulMuldense_1/Max:output:0dense_1/mul/y:output:0*
T0*
_output_shapes

:@X
dense_1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `At
dense_1/truediv_1RealDivdense_1/mul:z:0dense_1/truediv_1/y:output:0*
T0*
_output_shapes

:@R
dense_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3l
dense_1/addAddV2dense_1/truediv_1:z:0dense_1/add/y:output:0*
T0*
_output_shapes

:@L
dense_1/LogLogdense_1/add:z:0*
T0*
_output_shapes

:@X
dense_1/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?t
dense_1/truediv_2RealDivdense_1/Log:y:0dense_1/truediv_2/y:output:0*
T0*
_output_shapes

:@V
dense_1/RoundRounddense_1/truediv_2:z:0*
T0*
_output_shapes

:@T
dense_1/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @j
dense_1/Pow_1Powdense_1/Pow_1/x:output:0dense_1/Round:y:0*
T0*
_output_shapes

:@R
dense_1/Abs_1Absdense_1/truediv:z:0*
T0*
_output_shapes

:@@k
dense_1/truediv_3RealDivdense_1/Abs_1:y:0dense_1/Pow_1:z:0*
T0*
_output_shapes

:@@T
dense_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?p
dense_1/add_1AddV2dense_1/truediv_3:z:0dense_1/add_1/y:output:0*
T0*
_output_shapes

:@@R
dense_1/FloorFloordense_1/add_1:z:0*
T0*
_output_shapes

:@@S
dense_1/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@i
dense_1/LessLessdense_1/Floor:y:0dense_1/Less/y:output:0*
T0*
_output_shapes

:@@R
dense_1/SignSigndense_1/truediv:z:0*
T0*
_output_shapes

:@@x
'dense_1/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   \
dense_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/ones_likeFill0dense_1/ones_like/Shape/shape_as_tensor:output:0 dense_1/ones_like/Const:output:0*
T0*
_output_shapes

:@@T
dense_1/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `As
dense_1/mul_1Muldense_1/ones_like:output:0dense_1/mul_1/y:output:0*
T0*
_output_shapes

:@@X
dense_1/truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @v
dense_1/truediv_4RealDivdense_1/mul_1:z:0dense_1/truediv_4/y:output:0*
T0*
_output_shapes

:@@
dense_1/SelectV2SelectV2dense_1/Less:z:0dense_1/Floor:y:0dense_1/truediv_4:z:0*
T0*
_output_shapes

:@@j
dense_1/mul_2Muldense_1/Sign:y:0dense_1/SelectV2:output:0*
T0*
_output_shapes

:@@e
dense_1/Mul_3Muldense_1/truediv:z:0dense_1/mul_2:z:0*
T0*
_output_shapes

:@@h
dense_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/MeanMeandense_1/Mul_3:z:0'dense_1/Mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(c
dense_1/Mul_4Muldense_1/mul_2:z:0dense_1/mul_2:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_1Meandense_1/Mul_4:z:0)dense_1/Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(T
dense_1/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_1/add_2AddV2dense_1/Mean_1:output:0dense_1/add_2/y:output:0*
T0*
_output_shapes

:@o
dense_1/truediv_5RealDivdense_1/Mean:output:0dense_1/add_2:z:0*
T0*
_output_shapes

:@T
dense_1/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3p
dense_1/add_3AddV2dense_1/truediv_5:z:0dense_1/add_3/y:output:0*
T0*
_output_shapes

:@P
dense_1/Log_1Logdense_1/add_3:z:0*
T0*
_output_shapes

:@X
dense_1/truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?v
dense_1/truediv_6RealDivdense_1/Log_1:y:0dense_1/truediv_6/y:output:0*
T0*
_output_shapes

:@X
dense_1/Round_1Rounddense_1/truediv_6:z:0*
T0*
_output_shapes

:@T
dense_1/Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_1/Pow_2Powdense_1/Pow_2/x:output:0dense_1/Round_1:y:0*
T0*
_output_shapes

:@R
dense_1/Abs_2Absdense_1/truediv:z:0*
T0*
_output_shapes

:@@k
dense_1/truediv_7RealDivdense_1/Abs_2:y:0dense_1/Pow_2:z:0*
T0*
_output_shapes

:@@T
dense_1/add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?p
dense_1/add_4AddV2dense_1/truediv_7:z:0dense_1/add_4/y:output:0*
T0*
_output_shapes

:@@T
dense_1/Floor_1Floordense_1/add_4:z:0*
T0*
_output_shapes

:@@U
dense_1/Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_1/Less_1Lessdense_1/Floor_1:y:0dense_1/Less_1/y:output:0*
T0*
_output_shapes

:@@T
dense_1/Sign_1Signdense_1/truediv:z:0*
T0*
_output_shapes

:@@z
)dense_1/ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   ^
dense_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/ones_like_1Fill2dense_1/ones_like_1/Shape/shape_as_tensor:output:0"dense_1/ones_like_1/Const:output:0*
T0*
_output_shapes

:@@T
dense_1/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Au
dense_1/mul_5Muldense_1/ones_like_1:output:0dense_1/mul_5/y:output:0*
T0*
_output_shapes

:@@X
dense_1/truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @v
dense_1/truediv_8RealDivdense_1/mul_5:z:0dense_1/truediv_8/y:output:0*
T0*
_output_shapes

:@@
dense_1/SelectV2_1SelectV2dense_1/Less_1:z:0dense_1/Floor_1:y:0dense_1/truediv_8:z:0*
T0*
_output_shapes

:@@n
dense_1/mul_6Muldense_1/Sign_1:y:0dense_1/SelectV2_1:output:0*
T0*
_output_shapes

:@@e
dense_1/Mul_7Muldense_1/truediv:z:0dense_1/mul_6:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_2Meandense_1/Mul_7:z:0)dense_1/Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(c
dense_1/Mul_8Muldense_1/mul_6:z:0dense_1/mul_6:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_3Meandense_1/Mul_8:z:0)dense_1/Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(T
dense_1/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_1/add_5AddV2dense_1/Mean_3:output:0dense_1/add_5/y:output:0*
T0*
_output_shapes

:@q
dense_1/truediv_9RealDivdense_1/Mean_2:output:0dense_1/add_5:z:0*
T0*
_output_shapes

:@T
dense_1/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3p
dense_1/add_6AddV2dense_1/truediv_9:z:0dense_1/add_6/y:output:0*
T0*
_output_shapes

:@P
dense_1/Log_2Logdense_1/add_6:z:0*
T0*
_output_shapes

:@Y
dense_1/truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_1/truediv_10RealDivdense_1/Log_2:y:0dense_1/truediv_10/y:output:0*
T0*
_output_shapes

:@Y
dense_1/Round_2Rounddense_1/truediv_10:z:0*
T0*
_output_shapes

:@T
dense_1/Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_1/Pow_3Powdense_1/Pow_3/x:output:0dense_1/Round_2:y:0*
T0*
_output_shapes

:@R
dense_1/Abs_3Absdense_1/truediv:z:0*
T0*
_output_shapes

:@@l
dense_1/truediv_11RealDivdense_1/Abs_3:y:0dense_1/Pow_3:z:0*
T0*
_output_shapes

:@@T
dense_1/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?q
dense_1/add_7AddV2dense_1/truediv_11:z:0dense_1/add_7/y:output:0*
T0*
_output_shapes

:@@T
dense_1/Floor_2Floordense_1/add_7:z:0*
T0*
_output_shapes

:@@U
dense_1/Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_1/Less_2Lessdense_1/Floor_2:y:0dense_1/Less_2/y:output:0*
T0*
_output_shapes

:@@T
dense_1/Sign_2Signdense_1/truediv:z:0*
T0*
_output_shapes

:@@z
)dense_1/ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   ^
dense_1/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/ones_like_2Fill2dense_1/ones_like_2/Shape/shape_as_tensor:output:0"dense_1/ones_like_2/Const:output:0*
T0*
_output_shapes

:@@T
dense_1/mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Au
dense_1/mul_9Muldense_1/ones_like_2:output:0dense_1/mul_9/y:output:0*
T0*
_output_shapes

:@@Y
dense_1/truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @x
dense_1/truediv_12RealDivdense_1/mul_9:z:0dense_1/truediv_12/y:output:0*
T0*
_output_shapes

:@@
dense_1/SelectV2_2SelectV2dense_1/Less_2:z:0dense_1/Floor_2:y:0dense_1/truediv_12:z:0*
T0*
_output_shapes

:@@o
dense_1/mul_10Muldense_1/Sign_2:y:0dense_1/SelectV2_2:output:0*
T0*
_output_shapes

:@@g
dense_1/Mul_11Muldense_1/truediv:z:0dense_1/mul_10:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_4Meandense_1/Mul_11:z:0)dense_1/Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(f
dense_1/Mul_12Muldense_1/mul_10:z:0dense_1/mul_10:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_5Meandense_1/Mul_12:z:0)dense_1/Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(T
dense_1/add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_1/add_8AddV2dense_1/Mean_5:output:0dense_1/add_8/y:output:0*
T0*
_output_shapes

:@r
dense_1/truediv_13RealDivdense_1/Mean_4:output:0dense_1/add_8:z:0*
T0*
_output_shapes

:@T
dense_1/add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3q
dense_1/add_9AddV2dense_1/truediv_13:z:0dense_1/add_9/y:output:0*
T0*
_output_shapes

:@P
dense_1/Log_3Logdense_1/add_9:z:0*
T0*
_output_shapes

:@Y
dense_1/truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_1/truediv_14RealDivdense_1/Log_3:y:0dense_1/truediv_14/y:output:0*
T0*
_output_shapes

:@Y
dense_1/Round_3Rounddense_1/truediv_14:z:0*
T0*
_output_shapes

:@T
dense_1/Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_1/Pow_4Powdense_1/Pow_4/x:output:0dense_1/Round_3:y:0*
T0*
_output_shapes

:@R
dense_1/Abs_4Absdense_1/truediv:z:0*
T0*
_output_shapes

:@@l
dense_1/truediv_15RealDivdense_1/Abs_4:y:0dense_1/Pow_4:z:0*
T0*
_output_shapes

:@@U
dense_1/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?s
dense_1/add_10AddV2dense_1/truediv_15:z:0dense_1/add_10/y:output:0*
T0*
_output_shapes

:@@U
dense_1/Floor_3Floordense_1/add_10:z:0*
T0*
_output_shapes

:@@U
dense_1/Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_1/Less_3Lessdense_1/Floor_3:y:0dense_1/Less_3/y:output:0*
T0*
_output_shapes

:@@T
dense_1/Sign_3Signdense_1/truediv:z:0*
T0*
_output_shapes

:@@z
)dense_1/ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   ^
dense_1/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/ones_like_3Fill2dense_1/ones_like_3/Shape/shape_as_tensor:output:0"dense_1/ones_like_3/Const:output:0*
T0*
_output_shapes

:@@U
dense_1/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Aw
dense_1/mul_13Muldense_1/ones_like_3:output:0dense_1/mul_13/y:output:0*
T0*
_output_shapes

:@@Y
dense_1/truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @y
dense_1/truediv_16RealDivdense_1/mul_13:z:0dense_1/truediv_16/y:output:0*
T0*
_output_shapes

:@@
dense_1/SelectV2_3SelectV2dense_1/Less_3:z:0dense_1/Floor_3:y:0dense_1/truediv_16:z:0*
T0*
_output_shapes

:@@o
dense_1/mul_14Muldense_1/Sign_3:y:0dense_1/SelectV2_3:output:0*
T0*
_output_shapes

:@@g
dense_1/Mul_15Muldense_1/truediv:z:0dense_1/mul_14:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_6Meandense_1/Mul_15:z:0)dense_1/Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(f
dense_1/Mul_16Muldense_1/mul_14:z:0dense_1/mul_14:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_7Meandense_1/Mul_16:z:0)dense_1/Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(U
dense_1/add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3t
dense_1/add_11AddV2dense_1/Mean_7:output:0dense_1/add_11/y:output:0*
T0*
_output_shapes

:@s
dense_1/truediv_17RealDivdense_1/Mean_6:output:0dense_1/add_11:z:0*
T0*
_output_shapes

:@U
dense_1/add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3s
dense_1/add_12AddV2dense_1/truediv_17:z:0dense_1/add_12/y:output:0*
T0*
_output_shapes

:@Q
dense_1/Log_4Logdense_1/add_12:z:0*
T0*
_output_shapes

:@Y
dense_1/truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_1/truediv_18RealDivdense_1/Log_4:y:0dense_1/truediv_18/y:output:0*
T0*
_output_shapes

:@Y
dense_1/Round_4Rounddense_1/truediv_18:z:0*
T0*
_output_shapes

:@T
dense_1/Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_1/Pow_5Powdense_1/Pow_5/x:output:0dense_1/Round_4:y:0*
T0*
_output_shapes

:@R
dense_1/Abs_5Absdense_1/truediv:z:0*
T0*
_output_shapes

:@@l
dense_1/truediv_19RealDivdense_1/Abs_5:y:0dense_1/Pow_5:z:0*
T0*
_output_shapes

:@@U
dense_1/add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?s
dense_1/add_13AddV2dense_1/truediv_19:z:0dense_1/add_13/y:output:0*
T0*
_output_shapes

:@@U
dense_1/Floor_4Floordense_1/add_13:z:0*
T0*
_output_shapes

:@@U
dense_1/Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_1/Less_4Lessdense_1/Floor_4:y:0dense_1/Less_4/y:output:0*
T0*
_output_shapes

:@@T
dense_1/Sign_4Signdense_1/truediv:z:0*
T0*
_output_shapes

:@@z
)dense_1/ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   ^
dense_1/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_1/ones_like_4Fill2dense_1/ones_like_4/Shape/shape_as_tensor:output:0"dense_1/ones_like_4/Const:output:0*
T0*
_output_shapes

:@@U
dense_1/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Aw
dense_1/mul_17Muldense_1/ones_like_4:output:0dense_1/mul_17/y:output:0*
T0*
_output_shapes

:@@Y
dense_1/truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @y
dense_1/truediv_20RealDivdense_1/mul_17:z:0dense_1/truediv_20/y:output:0*
T0*
_output_shapes

:@@
dense_1/SelectV2_4SelectV2dense_1/Less_4:z:0dense_1/Floor_4:y:0dense_1/truediv_20:z:0*
T0*
_output_shapes

:@@o
dense_1/mul_18Muldense_1/Sign_4:y:0dense_1/SelectV2_4:output:0*
T0*
_output_shapes

:@@g
dense_1/Mul_19Muldense_1/truediv:z:0dense_1/mul_18:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_8Meandense_1/Mul_19:z:0)dense_1/Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(f
dense_1/Mul_20Muldense_1/mul_18:z:0dense_1/mul_18:z:0*
T0*
_output_shapes

:@@j
 dense_1/Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Mean_9Meandense_1/Mul_20:z:0)dense_1/Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(U
dense_1/add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3t
dense_1/add_14AddV2dense_1/Mean_9:output:0dense_1/add_14/y:output:0*
T0*
_output_shapes

:@s
dense_1/truediv_21RealDivdense_1/Mean_8:output:0dense_1/add_14:z:0*
T0*
_output_shapes

:@U
dense_1/add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3s
dense_1/add_15AddV2dense_1/truediv_21:z:0dense_1/add_15/y:output:0*
T0*
_output_shapes

:@Q
dense_1/Log_5Logdense_1/add_15:z:0*
T0*
_output_shapes

:@Y
dense_1/truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_1/truediv_22RealDivdense_1/Log_5:y:0dense_1/truediv_22/y:output:0*
T0*
_output_shapes

:@Y
dense_1/Round_5Rounddense_1/truediv_22:z:0*
T0*
_output_shapes

:@T
dense_1/Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_1/Pow_6Powdense_1/Pow_6/x:output:0dense_1/Round_5:y:0*
T0*
_output_shapes

:@U
dense_1/mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Al
dense_1/mul_21Muldense_1/Pow_6:z:0dense_1/mul_21/y:output:0*
T0*
_output_shapes

:@e
dense_1/mul_22Muldense_1/Cast:y:0dense_1/truediv:z:0*
T0*
_output_shapes

:@@d
dense_1/mul_23Muldense_1/Cast:y:0dense_1/mul_18:z:0*
T0*
_output_shapes

:@@Y
dense_1/truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ay
dense_1/truediv_23RealDivdense_1/mul_23:z:0dense_1/truediv_23/y:output:0*
T0*
_output_shapes

:@@j
dense_1/mul_24Muldense_1/mul_21:z:0dense_1/truediv_23:z:0*
T0*
_output_shapes

:@@O
dense_1/NegNegdense_1/mul_22:z:0*
T0*
_output_shapes

:@@e
dense_1/add_16AddV2dense_1/Neg:y:0dense_1/mul_24:z:0*
T0*
_output_shapes

:@@U
dense_1/mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
dense_1/mul_25Muldense_1/mul_25/x:output:0dense_1/add_16:z:0*
T0*
_output_shapes

:@@a
dense_1/StopGradientStopGradientdense_1/mul_25:z:0*
T0*
_output_shapes

:@@s
dense_1/add_17AddV2dense_1/mul_22:z:0dense_1/StopGradient:output:0*
T0*
_output_shapes

:@@o
dense_1/MatMulMatMulre_lu/add_2:z:0dense_1/add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
dense_1/Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :Q
dense_1/Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : i
dense_1/Pow_7Powdense_1/Pow_7/x:output:0dense_1/Pow_7/y:output:0*
T0*
_output_shapes
: Y
dense_1/Cast_1Castdense_1/Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: v
dense_1/ReadVariableOp_1ReadVariableOp!dense_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0U
dense_1/mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aw
dense_1/mul_26Mul dense_1/ReadVariableOp_1:value:0dense_1/mul_26/y:output:0*
T0*
_output_shapes
:@j
dense_1/truediv_24RealDivdense_1/mul_26:z:0dense_1/Cast_1:y:0*
T0*
_output_shapes
:@Q
dense_1/Neg_1Negdense_1/truediv_24:z:0*
T0*
_output_shapes
:@U
dense_1/Round_6Rounddense_1/truediv_24:z:0*
T0*
_output_shapes
:@d
dense_1/add_18AddV2dense_1/Neg_1:y:0dense_1/Round_6:y:0*
T0*
_output_shapes
:@_
dense_1/StopGradient_1StopGradientdense_1/add_18:z:0*
T0*
_output_shapes
:@u
dense_1/add_19AddV2dense_1/truediv_24:z:0dense_1/StopGradient_1:output:0*
T0*
_output_shapes
:@d
dense_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
dense_1/clip_by_value/MinimumMinimumdense_1/add_19:z:0(dense_1/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:@\
dense_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á
dense_1/clip_by_valueMaximum!dense_1/clip_by_value/Minimum:z:0 dense_1/clip_by_value/y:output:0*
T0*
_output_shapes
:@i
dense_1/mul_27Muldense_1/Cast_1:y:0dense_1/clip_by_value:z:0*
T0*
_output_shapes
:@Y
dense_1/truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Au
dense_1/truediv_25RealDivdense_1/mul_27:z:0dense_1/truediv_25/y:output:0*
T0*
_output_shapes
:@U
dense_1/mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
dense_1/mul_28Muldense_1/mul_28/x:output:0dense_1/truediv_25:z:0*
T0*
_output_shapes
:@v
dense_1/ReadVariableOp_2ReadVariableOp!dense_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0[
dense_1/Neg_2Neg dense_1/ReadVariableOp_2:value:0*
T0*
_output_shapes
:@c
dense_1/add_20AddV2dense_1/Neg_2:y:0dense_1/mul_28:z:0*
T0*
_output_shapes
:@U
dense_1/mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
dense_1/mul_29Muldense_1/mul_29/x:output:0dense_1/add_20:z:0*
T0*
_output_shapes
:@_
dense_1/StopGradient_2StopGradientdense_1/mul_29:z:0*
T0*
_output_shapes
:@v
dense_1/ReadVariableOp_3ReadVariableOp!dense_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0
dense_1/add_21AddV2 dense_1/ReadVariableOp_3:value:0dense_1/StopGradient_2:output:0*
T0*
_output_shapes
:@z
dense_1/BiasAddBiasAdddense_1/MatMul:product:0dense_1/add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@ª
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0¶
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@¡
%batch_normalization_1/batchnorm/mul_1Muldense_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0´
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@¦
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0´
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@´
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
dropout_1/IdentityIdentity)batch_normalization_1/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
re_lu_1/SignSigndropout_1/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@V
re_lu_1/AbsAbsre_lu_1/Sign:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
re_lu_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
re_lu_1/subSubre_lu_1/sub/x:output:0re_lu_1/Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
re_lu_1/addAddV2re_lu_1/Sign:y:0re_lu_1/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
re_lu_1/TanhTanhdropout_1/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@V
re_lu_1/NegNegre_lu_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
re_lu_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
re_lu_1/mulMulre_lu_1/mul/x:output:0re_lu_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
re_lu_1/add_1AddV2re_lu_1/Neg:y:0re_lu_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
re_lu_1/StopGradientStopGradientre_lu_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
re_lu_1/add_2AddV2re_lu_1/Tanh:y:0re_lu_1/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@O
dense_2/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :O
dense_2/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : c
dense_2/PowPowdense_2/Pow/x:output:0dense_2/Pow/y:output:0*
T0*
_output_shapes
: U
dense_2/CastCastdense_2/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: v
dense_2/ReadVariableOpReadVariableOpdense_2_readvariableop_resource*
_output_shapes

:@@*
dtype0u
dense_2/truedivRealDivdense_2/ReadVariableOp:value:0dense_2/Cast:y:0*
T0*
_output_shapes

:@@P
dense_2/AbsAbsdense_2/truediv:z:0*
T0*
_output_shapes

:@@g
dense_2/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/MaxMaxdense_2/Abs:y:0&dense_2/Max/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(R
dense_2/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @i
dense_2/mulMuldense_2/Max:output:0dense_2/mul/y:output:0*
T0*
_output_shapes

:@X
dense_2/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `At
dense_2/truediv_1RealDivdense_2/mul:z:0dense_2/truediv_1/y:output:0*
T0*
_output_shapes

:@R
dense_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3l
dense_2/addAddV2dense_2/truediv_1:z:0dense_2/add/y:output:0*
T0*
_output_shapes

:@L
dense_2/LogLogdense_2/add:z:0*
T0*
_output_shapes

:@X
dense_2/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?t
dense_2/truediv_2RealDivdense_2/Log:y:0dense_2/truediv_2/y:output:0*
T0*
_output_shapes

:@V
dense_2/RoundRounddense_2/truediv_2:z:0*
T0*
_output_shapes

:@T
dense_2/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @j
dense_2/Pow_1Powdense_2/Pow_1/x:output:0dense_2/Round:y:0*
T0*
_output_shapes

:@R
dense_2/Abs_1Absdense_2/truediv:z:0*
T0*
_output_shapes

:@@k
dense_2/truediv_3RealDivdense_2/Abs_1:y:0dense_2/Pow_1:z:0*
T0*
_output_shapes

:@@T
dense_2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?p
dense_2/add_1AddV2dense_2/truediv_3:z:0dense_2/add_1/y:output:0*
T0*
_output_shapes

:@@R
dense_2/FloorFloordense_2/add_1:z:0*
T0*
_output_shapes

:@@S
dense_2/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@i
dense_2/LessLessdense_2/Floor:y:0dense_2/Less/y:output:0*
T0*
_output_shapes

:@@R
dense_2/SignSigndense_2/truediv:z:0*
T0*
_output_shapes

:@@x
'dense_2/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   \
dense_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_2/ones_likeFill0dense_2/ones_like/Shape/shape_as_tensor:output:0 dense_2/ones_like/Const:output:0*
T0*
_output_shapes

:@@T
dense_2/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `As
dense_2/mul_1Muldense_2/ones_like:output:0dense_2/mul_1/y:output:0*
T0*
_output_shapes

:@@X
dense_2/truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @v
dense_2/truediv_4RealDivdense_2/mul_1:z:0dense_2/truediv_4/y:output:0*
T0*
_output_shapes

:@@
dense_2/SelectV2SelectV2dense_2/Less:z:0dense_2/Floor:y:0dense_2/truediv_4:z:0*
T0*
_output_shapes

:@@j
dense_2/mul_2Muldense_2/Sign:y:0dense_2/SelectV2:output:0*
T0*
_output_shapes

:@@e
dense_2/Mul_3Muldense_2/truediv:z:0dense_2/mul_2:z:0*
T0*
_output_shapes

:@@h
dense_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/MeanMeandense_2/Mul_3:z:0'dense_2/Mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(c
dense_2/Mul_4Muldense_2/mul_2:z:0dense_2/mul_2:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_1Meandense_2/Mul_4:z:0)dense_2/Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(T
dense_2/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_2/add_2AddV2dense_2/Mean_1:output:0dense_2/add_2/y:output:0*
T0*
_output_shapes

:@o
dense_2/truediv_5RealDivdense_2/Mean:output:0dense_2/add_2:z:0*
T0*
_output_shapes

:@T
dense_2/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3p
dense_2/add_3AddV2dense_2/truediv_5:z:0dense_2/add_3/y:output:0*
T0*
_output_shapes

:@P
dense_2/Log_1Logdense_2/add_3:z:0*
T0*
_output_shapes

:@X
dense_2/truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?v
dense_2/truediv_6RealDivdense_2/Log_1:y:0dense_2/truediv_6/y:output:0*
T0*
_output_shapes

:@X
dense_2/Round_1Rounddense_2/truediv_6:z:0*
T0*
_output_shapes

:@T
dense_2/Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_2/Pow_2Powdense_2/Pow_2/x:output:0dense_2/Round_1:y:0*
T0*
_output_shapes

:@R
dense_2/Abs_2Absdense_2/truediv:z:0*
T0*
_output_shapes

:@@k
dense_2/truediv_7RealDivdense_2/Abs_2:y:0dense_2/Pow_2:z:0*
T0*
_output_shapes

:@@T
dense_2/add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?p
dense_2/add_4AddV2dense_2/truediv_7:z:0dense_2/add_4/y:output:0*
T0*
_output_shapes

:@@T
dense_2/Floor_1Floordense_2/add_4:z:0*
T0*
_output_shapes

:@@U
dense_2/Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_2/Less_1Lessdense_2/Floor_1:y:0dense_2/Less_1/y:output:0*
T0*
_output_shapes

:@@T
dense_2/Sign_1Signdense_2/truediv:z:0*
T0*
_output_shapes

:@@z
)dense_2/ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   ^
dense_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_2/ones_like_1Fill2dense_2/ones_like_1/Shape/shape_as_tensor:output:0"dense_2/ones_like_1/Const:output:0*
T0*
_output_shapes

:@@T
dense_2/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Au
dense_2/mul_5Muldense_2/ones_like_1:output:0dense_2/mul_5/y:output:0*
T0*
_output_shapes

:@@X
dense_2/truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @v
dense_2/truediv_8RealDivdense_2/mul_5:z:0dense_2/truediv_8/y:output:0*
T0*
_output_shapes

:@@
dense_2/SelectV2_1SelectV2dense_2/Less_1:z:0dense_2/Floor_1:y:0dense_2/truediv_8:z:0*
T0*
_output_shapes

:@@n
dense_2/mul_6Muldense_2/Sign_1:y:0dense_2/SelectV2_1:output:0*
T0*
_output_shapes

:@@e
dense_2/Mul_7Muldense_2/truediv:z:0dense_2/mul_6:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_2Meandense_2/Mul_7:z:0)dense_2/Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(c
dense_2/Mul_8Muldense_2/mul_6:z:0dense_2/mul_6:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_3Meandense_2/Mul_8:z:0)dense_2/Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(T
dense_2/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_2/add_5AddV2dense_2/Mean_3:output:0dense_2/add_5/y:output:0*
T0*
_output_shapes

:@q
dense_2/truediv_9RealDivdense_2/Mean_2:output:0dense_2/add_5:z:0*
T0*
_output_shapes

:@T
dense_2/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3p
dense_2/add_6AddV2dense_2/truediv_9:z:0dense_2/add_6/y:output:0*
T0*
_output_shapes

:@P
dense_2/Log_2Logdense_2/add_6:z:0*
T0*
_output_shapes

:@Y
dense_2/truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_2/truediv_10RealDivdense_2/Log_2:y:0dense_2/truediv_10/y:output:0*
T0*
_output_shapes

:@Y
dense_2/Round_2Rounddense_2/truediv_10:z:0*
T0*
_output_shapes

:@T
dense_2/Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_2/Pow_3Powdense_2/Pow_3/x:output:0dense_2/Round_2:y:0*
T0*
_output_shapes

:@R
dense_2/Abs_3Absdense_2/truediv:z:0*
T0*
_output_shapes

:@@l
dense_2/truediv_11RealDivdense_2/Abs_3:y:0dense_2/Pow_3:z:0*
T0*
_output_shapes

:@@T
dense_2/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?q
dense_2/add_7AddV2dense_2/truediv_11:z:0dense_2/add_7/y:output:0*
T0*
_output_shapes

:@@T
dense_2/Floor_2Floordense_2/add_7:z:0*
T0*
_output_shapes

:@@U
dense_2/Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_2/Less_2Lessdense_2/Floor_2:y:0dense_2/Less_2/y:output:0*
T0*
_output_shapes

:@@T
dense_2/Sign_2Signdense_2/truediv:z:0*
T0*
_output_shapes

:@@z
)dense_2/ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   ^
dense_2/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_2/ones_like_2Fill2dense_2/ones_like_2/Shape/shape_as_tensor:output:0"dense_2/ones_like_2/Const:output:0*
T0*
_output_shapes

:@@T
dense_2/mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Au
dense_2/mul_9Muldense_2/ones_like_2:output:0dense_2/mul_9/y:output:0*
T0*
_output_shapes

:@@Y
dense_2/truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @x
dense_2/truediv_12RealDivdense_2/mul_9:z:0dense_2/truediv_12/y:output:0*
T0*
_output_shapes

:@@
dense_2/SelectV2_2SelectV2dense_2/Less_2:z:0dense_2/Floor_2:y:0dense_2/truediv_12:z:0*
T0*
_output_shapes

:@@o
dense_2/mul_10Muldense_2/Sign_2:y:0dense_2/SelectV2_2:output:0*
T0*
_output_shapes

:@@g
dense_2/Mul_11Muldense_2/truediv:z:0dense_2/mul_10:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_4Meandense_2/Mul_11:z:0)dense_2/Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(f
dense_2/Mul_12Muldense_2/mul_10:z:0dense_2/mul_10:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_5Meandense_2/Mul_12:z:0)dense_2/Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(T
dense_2/add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_2/add_8AddV2dense_2/Mean_5:output:0dense_2/add_8/y:output:0*
T0*
_output_shapes

:@r
dense_2/truediv_13RealDivdense_2/Mean_4:output:0dense_2/add_8:z:0*
T0*
_output_shapes

:@T
dense_2/add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3q
dense_2/add_9AddV2dense_2/truediv_13:z:0dense_2/add_9/y:output:0*
T0*
_output_shapes

:@P
dense_2/Log_3Logdense_2/add_9:z:0*
T0*
_output_shapes

:@Y
dense_2/truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_2/truediv_14RealDivdense_2/Log_3:y:0dense_2/truediv_14/y:output:0*
T0*
_output_shapes

:@Y
dense_2/Round_3Rounddense_2/truediv_14:z:0*
T0*
_output_shapes

:@T
dense_2/Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_2/Pow_4Powdense_2/Pow_4/x:output:0dense_2/Round_3:y:0*
T0*
_output_shapes

:@R
dense_2/Abs_4Absdense_2/truediv:z:0*
T0*
_output_shapes

:@@l
dense_2/truediv_15RealDivdense_2/Abs_4:y:0dense_2/Pow_4:z:0*
T0*
_output_shapes

:@@U
dense_2/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?s
dense_2/add_10AddV2dense_2/truediv_15:z:0dense_2/add_10/y:output:0*
T0*
_output_shapes

:@@U
dense_2/Floor_3Floordense_2/add_10:z:0*
T0*
_output_shapes

:@@U
dense_2/Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_2/Less_3Lessdense_2/Floor_3:y:0dense_2/Less_3/y:output:0*
T0*
_output_shapes

:@@T
dense_2/Sign_3Signdense_2/truediv:z:0*
T0*
_output_shapes

:@@z
)dense_2/ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   ^
dense_2/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_2/ones_like_3Fill2dense_2/ones_like_3/Shape/shape_as_tensor:output:0"dense_2/ones_like_3/Const:output:0*
T0*
_output_shapes

:@@U
dense_2/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Aw
dense_2/mul_13Muldense_2/ones_like_3:output:0dense_2/mul_13/y:output:0*
T0*
_output_shapes

:@@Y
dense_2/truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @y
dense_2/truediv_16RealDivdense_2/mul_13:z:0dense_2/truediv_16/y:output:0*
T0*
_output_shapes

:@@
dense_2/SelectV2_3SelectV2dense_2/Less_3:z:0dense_2/Floor_3:y:0dense_2/truediv_16:z:0*
T0*
_output_shapes

:@@o
dense_2/mul_14Muldense_2/Sign_3:y:0dense_2/SelectV2_3:output:0*
T0*
_output_shapes

:@@g
dense_2/Mul_15Muldense_2/truediv:z:0dense_2/mul_14:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_6Meandense_2/Mul_15:z:0)dense_2/Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(f
dense_2/Mul_16Muldense_2/mul_14:z:0dense_2/mul_14:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_7Meandense_2/Mul_16:z:0)dense_2/Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(U
dense_2/add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3t
dense_2/add_11AddV2dense_2/Mean_7:output:0dense_2/add_11/y:output:0*
T0*
_output_shapes

:@s
dense_2/truediv_17RealDivdense_2/Mean_6:output:0dense_2/add_11:z:0*
T0*
_output_shapes

:@U
dense_2/add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3s
dense_2/add_12AddV2dense_2/truediv_17:z:0dense_2/add_12/y:output:0*
T0*
_output_shapes

:@Q
dense_2/Log_4Logdense_2/add_12:z:0*
T0*
_output_shapes

:@Y
dense_2/truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_2/truediv_18RealDivdense_2/Log_4:y:0dense_2/truediv_18/y:output:0*
T0*
_output_shapes

:@Y
dense_2/Round_4Rounddense_2/truediv_18:z:0*
T0*
_output_shapes

:@T
dense_2/Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_2/Pow_5Powdense_2/Pow_5/x:output:0dense_2/Round_4:y:0*
T0*
_output_shapes

:@R
dense_2/Abs_5Absdense_2/truediv:z:0*
T0*
_output_shapes

:@@l
dense_2/truediv_19RealDivdense_2/Abs_5:y:0dense_2/Pow_5:z:0*
T0*
_output_shapes

:@@U
dense_2/add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?s
dense_2/add_13AddV2dense_2/truediv_19:z:0dense_2/add_13/y:output:0*
T0*
_output_shapes

:@@U
dense_2/Floor_4Floordense_2/add_13:z:0*
T0*
_output_shapes

:@@U
dense_2/Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_2/Less_4Lessdense_2/Floor_4:y:0dense_2/Less_4/y:output:0*
T0*
_output_shapes

:@@T
dense_2/Sign_4Signdense_2/truediv:z:0*
T0*
_output_shapes

:@@z
)dense_2/ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   ^
dense_2/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_2/ones_like_4Fill2dense_2/ones_like_4/Shape/shape_as_tensor:output:0"dense_2/ones_like_4/Const:output:0*
T0*
_output_shapes

:@@U
dense_2/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Aw
dense_2/mul_17Muldense_2/ones_like_4:output:0dense_2/mul_17/y:output:0*
T0*
_output_shapes

:@@Y
dense_2/truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @y
dense_2/truediv_20RealDivdense_2/mul_17:z:0dense_2/truediv_20/y:output:0*
T0*
_output_shapes

:@@
dense_2/SelectV2_4SelectV2dense_2/Less_4:z:0dense_2/Floor_4:y:0dense_2/truediv_20:z:0*
T0*
_output_shapes

:@@o
dense_2/mul_18Muldense_2/Sign_4:y:0dense_2/SelectV2_4:output:0*
T0*
_output_shapes

:@@g
dense_2/Mul_19Muldense_2/truediv:z:0dense_2/mul_18:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_8Meandense_2/Mul_19:z:0)dense_2/Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(f
dense_2/Mul_20Muldense_2/mul_18:z:0dense_2/mul_18:z:0*
T0*
_output_shapes

:@@j
 dense_2/Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Mean_9Meandense_2/Mul_20:z:0)dense_2/Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(U
dense_2/add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3t
dense_2/add_14AddV2dense_2/Mean_9:output:0dense_2/add_14/y:output:0*
T0*
_output_shapes

:@s
dense_2/truediv_21RealDivdense_2/Mean_8:output:0dense_2/add_14:z:0*
T0*
_output_shapes

:@U
dense_2/add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3s
dense_2/add_15AddV2dense_2/truediv_21:z:0dense_2/add_15/y:output:0*
T0*
_output_shapes

:@Q
dense_2/Log_5Logdense_2/add_15:z:0*
T0*
_output_shapes

:@Y
dense_2/truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_2/truediv_22RealDivdense_2/Log_5:y:0dense_2/truediv_22/y:output:0*
T0*
_output_shapes

:@Y
dense_2/Round_5Rounddense_2/truediv_22:z:0*
T0*
_output_shapes

:@T
dense_2/Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_2/Pow_6Powdense_2/Pow_6/x:output:0dense_2/Round_5:y:0*
T0*
_output_shapes

:@U
dense_2/mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Al
dense_2/mul_21Muldense_2/Pow_6:z:0dense_2/mul_21/y:output:0*
T0*
_output_shapes

:@e
dense_2/mul_22Muldense_2/Cast:y:0dense_2/truediv:z:0*
T0*
_output_shapes

:@@d
dense_2/mul_23Muldense_2/Cast:y:0dense_2/mul_18:z:0*
T0*
_output_shapes

:@@Y
dense_2/truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ay
dense_2/truediv_23RealDivdense_2/mul_23:z:0dense_2/truediv_23/y:output:0*
T0*
_output_shapes

:@@j
dense_2/mul_24Muldense_2/mul_21:z:0dense_2/truediv_23:z:0*
T0*
_output_shapes

:@@O
dense_2/NegNegdense_2/mul_22:z:0*
T0*
_output_shapes

:@@e
dense_2/add_16AddV2dense_2/Neg:y:0dense_2/mul_24:z:0*
T0*
_output_shapes

:@@U
dense_2/mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
dense_2/mul_25Muldense_2/mul_25/x:output:0dense_2/add_16:z:0*
T0*
_output_shapes

:@@a
dense_2/StopGradientStopGradientdense_2/mul_25:z:0*
T0*
_output_shapes

:@@s
dense_2/add_17AddV2dense_2/mul_22:z:0dense_2/StopGradient:output:0*
T0*
_output_shapes

:@@q
dense_2/MatMulMatMulre_lu_1/add_2:z:0dense_2/add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
dense_2/Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :Q
dense_2/Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : i
dense_2/Pow_7Powdense_2/Pow_7/x:output:0dense_2/Pow_7/y:output:0*
T0*
_output_shapes
: Y
dense_2/Cast_1Castdense_2/Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: v
dense_2/ReadVariableOp_1ReadVariableOp!dense_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0U
dense_2/mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aw
dense_2/mul_26Mul dense_2/ReadVariableOp_1:value:0dense_2/mul_26/y:output:0*
T0*
_output_shapes
:@j
dense_2/truediv_24RealDivdense_2/mul_26:z:0dense_2/Cast_1:y:0*
T0*
_output_shapes
:@Q
dense_2/Neg_1Negdense_2/truediv_24:z:0*
T0*
_output_shapes
:@U
dense_2/Round_6Rounddense_2/truediv_24:z:0*
T0*
_output_shapes
:@d
dense_2/add_18AddV2dense_2/Neg_1:y:0dense_2/Round_6:y:0*
T0*
_output_shapes
:@_
dense_2/StopGradient_1StopGradientdense_2/add_18:z:0*
T0*
_output_shapes
:@u
dense_2/add_19AddV2dense_2/truediv_24:z:0dense_2/StopGradient_1:output:0*
T0*
_output_shapes
:@d
dense_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
dense_2/clip_by_value/MinimumMinimumdense_2/add_19:z:0(dense_2/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:@\
dense_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á
dense_2/clip_by_valueMaximum!dense_2/clip_by_value/Minimum:z:0 dense_2/clip_by_value/y:output:0*
T0*
_output_shapes
:@i
dense_2/mul_27Muldense_2/Cast_1:y:0dense_2/clip_by_value:z:0*
T0*
_output_shapes
:@Y
dense_2/truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Au
dense_2/truediv_25RealDivdense_2/mul_27:z:0dense_2/truediv_25/y:output:0*
T0*
_output_shapes
:@U
dense_2/mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
dense_2/mul_28Muldense_2/mul_28/x:output:0dense_2/truediv_25:z:0*
T0*
_output_shapes
:@v
dense_2/ReadVariableOp_2ReadVariableOp!dense_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0[
dense_2/Neg_2Neg dense_2/ReadVariableOp_2:value:0*
T0*
_output_shapes
:@c
dense_2/add_20AddV2dense_2/Neg_2:y:0dense_2/mul_28:z:0*
T0*
_output_shapes
:@U
dense_2/mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
dense_2/mul_29Muldense_2/mul_29/x:output:0dense_2/add_20:z:0*
T0*
_output_shapes
:@_
dense_2/StopGradient_2StopGradientdense_2/mul_29:z:0*
T0*
_output_shapes
:@v
dense_2/ReadVariableOp_3ReadVariableOp!dense_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0
dense_2/add_21AddV2 dense_2/ReadVariableOp_3:value:0dense_2/StopGradient_2:output:0*
T0*
_output_shapes
:@z
dense_2/BiasAddBiasAdddense_2/MatMul:product:0dense_2/add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¹
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@ª
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0¶
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@¡
%batch_normalization_2/batchnorm/mul_1Muldense_2/BiasAdd:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0´
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@¦
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0´
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@´
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
dropout_2/IdentityIdentity)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
re_lu_2/SignSigndropout_2/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@V
re_lu_2/AbsAbsre_lu_2/Sign:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
re_lu_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
re_lu_2/subSubre_lu_2/sub/x:output:0re_lu_2/Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
re_lu_2/addAddV2re_lu_2/Sign:y:0re_lu_2/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
re_lu_2/TanhTanhdropout_2/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@V
re_lu_2/NegNegre_lu_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
re_lu_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
re_lu_2/mulMulre_lu_2/mul/x:output:0re_lu_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@j
re_lu_2/add_1AddV2re_lu_2/Neg:y:0re_lu_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
re_lu_2/StopGradientStopGradientre_lu_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
re_lu_2/add_2AddV2re_lu_2/Tanh:y:0re_lu_2/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@O
dense_3/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :O
dense_3/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : c
dense_3/PowPowdense_3/Pow/x:output:0dense_3/Pow/y:output:0*
T0*
_output_shapes
: U
dense_3/CastCastdense_3/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: v
dense_3/ReadVariableOpReadVariableOpdense_3_readvariableop_resource*
_output_shapes

:@*
dtype0u
dense_3/truedivRealDivdense_3/ReadVariableOp:value:0dense_3/Cast:y:0*
T0*
_output_shapes

:@P
dense_3/AbsAbsdense_3/truediv:z:0*
T0*
_output_shapes

:@g
dense_3/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/MaxMaxdense_3/Abs:y:0&dense_3/Max/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(R
dense_3/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @i
dense_3/mulMuldense_3/Max:output:0dense_3/mul/y:output:0*
T0*
_output_shapes

:X
dense_3/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `At
dense_3/truediv_1RealDivdense_3/mul:z:0dense_3/truediv_1/y:output:0*
T0*
_output_shapes

:R
dense_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3l
dense_3/addAddV2dense_3/truediv_1:z:0dense_3/add/y:output:0*
T0*
_output_shapes

:L
dense_3/LogLogdense_3/add:z:0*
T0*
_output_shapes

:X
dense_3/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?t
dense_3/truediv_2RealDivdense_3/Log:y:0dense_3/truediv_2/y:output:0*
T0*
_output_shapes

:V
dense_3/RoundRounddense_3/truediv_2:z:0*
T0*
_output_shapes

:T
dense_3/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @j
dense_3/Pow_1Powdense_3/Pow_1/x:output:0dense_3/Round:y:0*
T0*
_output_shapes

:R
dense_3/Abs_1Absdense_3/truediv:z:0*
T0*
_output_shapes

:@k
dense_3/truediv_3RealDivdense_3/Abs_1:y:0dense_3/Pow_1:z:0*
T0*
_output_shapes

:@T
dense_3/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?p
dense_3/add_1AddV2dense_3/truediv_3:z:0dense_3/add_1/y:output:0*
T0*
_output_shapes

:@R
dense_3/FloorFloordense_3/add_1:z:0*
T0*
_output_shapes

:@S
dense_3/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@i
dense_3/LessLessdense_3/Floor:y:0dense_3/Less/y:output:0*
T0*
_output_shapes

:@R
dense_3/SignSigndense_3/truediv:z:0*
T0*
_output_shapes

:@x
'dense_3/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      \
dense_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_3/ones_likeFill0dense_3/ones_like/Shape/shape_as_tensor:output:0 dense_3/ones_like/Const:output:0*
T0*
_output_shapes

:@T
dense_3/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `As
dense_3/mul_1Muldense_3/ones_like:output:0dense_3/mul_1/y:output:0*
T0*
_output_shapes

:@X
dense_3/truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @v
dense_3/truediv_4RealDivdense_3/mul_1:z:0dense_3/truediv_4/y:output:0*
T0*
_output_shapes

:@
dense_3/SelectV2SelectV2dense_3/Less:z:0dense_3/Floor:y:0dense_3/truediv_4:z:0*
T0*
_output_shapes

:@j
dense_3/mul_2Muldense_3/Sign:y:0dense_3/SelectV2:output:0*
T0*
_output_shapes

:@e
dense_3/Mul_3Muldense_3/truediv:z:0dense_3/mul_2:z:0*
T0*
_output_shapes

:@h
dense_3/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/MeanMeandense_3/Mul_3:z:0'dense_3/Mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(c
dense_3/Mul_4Muldense_3/mul_2:z:0dense_3/mul_2:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_1Meandense_3/Mul_4:z:0)dense_3/Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(T
dense_3/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_3/add_2AddV2dense_3/Mean_1:output:0dense_3/add_2/y:output:0*
T0*
_output_shapes

:o
dense_3/truediv_5RealDivdense_3/Mean:output:0dense_3/add_2:z:0*
T0*
_output_shapes

:T
dense_3/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3p
dense_3/add_3AddV2dense_3/truediv_5:z:0dense_3/add_3/y:output:0*
T0*
_output_shapes

:P
dense_3/Log_1Logdense_3/add_3:z:0*
T0*
_output_shapes

:X
dense_3/truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?v
dense_3/truediv_6RealDivdense_3/Log_1:y:0dense_3/truediv_6/y:output:0*
T0*
_output_shapes

:X
dense_3/Round_1Rounddense_3/truediv_6:z:0*
T0*
_output_shapes

:T
dense_3/Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_3/Pow_2Powdense_3/Pow_2/x:output:0dense_3/Round_1:y:0*
T0*
_output_shapes

:R
dense_3/Abs_2Absdense_3/truediv:z:0*
T0*
_output_shapes

:@k
dense_3/truediv_7RealDivdense_3/Abs_2:y:0dense_3/Pow_2:z:0*
T0*
_output_shapes

:@T
dense_3/add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?p
dense_3/add_4AddV2dense_3/truediv_7:z:0dense_3/add_4/y:output:0*
T0*
_output_shapes

:@T
dense_3/Floor_1Floordense_3/add_4:z:0*
T0*
_output_shapes

:@U
dense_3/Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_3/Less_1Lessdense_3/Floor_1:y:0dense_3/Less_1/y:output:0*
T0*
_output_shapes

:@T
dense_3/Sign_1Signdense_3/truediv:z:0*
T0*
_output_shapes

:@z
)dense_3/ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      ^
dense_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_3/ones_like_1Fill2dense_3/ones_like_1/Shape/shape_as_tensor:output:0"dense_3/ones_like_1/Const:output:0*
T0*
_output_shapes

:@T
dense_3/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Au
dense_3/mul_5Muldense_3/ones_like_1:output:0dense_3/mul_5/y:output:0*
T0*
_output_shapes

:@X
dense_3/truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @v
dense_3/truediv_8RealDivdense_3/mul_5:z:0dense_3/truediv_8/y:output:0*
T0*
_output_shapes

:@
dense_3/SelectV2_1SelectV2dense_3/Less_1:z:0dense_3/Floor_1:y:0dense_3/truediv_8:z:0*
T0*
_output_shapes

:@n
dense_3/mul_6Muldense_3/Sign_1:y:0dense_3/SelectV2_1:output:0*
T0*
_output_shapes

:@e
dense_3/Mul_7Muldense_3/truediv:z:0dense_3/mul_6:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_2Meandense_3/Mul_7:z:0)dense_3/Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(c
dense_3/Mul_8Muldense_3/mul_6:z:0dense_3/mul_6:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_3Meandense_3/Mul_8:z:0)dense_3/Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(T
dense_3/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_3/add_5AddV2dense_3/Mean_3:output:0dense_3/add_5/y:output:0*
T0*
_output_shapes

:q
dense_3/truediv_9RealDivdense_3/Mean_2:output:0dense_3/add_5:z:0*
T0*
_output_shapes

:T
dense_3/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3p
dense_3/add_6AddV2dense_3/truediv_9:z:0dense_3/add_6/y:output:0*
T0*
_output_shapes

:P
dense_3/Log_2Logdense_3/add_6:z:0*
T0*
_output_shapes

:Y
dense_3/truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_3/truediv_10RealDivdense_3/Log_2:y:0dense_3/truediv_10/y:output:0*
T0*
_output_shapes

:Y
dense_3/Round_2Rounddense_3/truediv_10:z:0*
T0*
_output_shapes

:T
dense_3/Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_3/Pow_3Powdense_3/Pow_3/x:output:0dense_3/Round_2:y:0*
T0*
_output_shapes

:R
dense_3/Abs_3Absdense_3/truediv:z:0*
T0*
_output_shapes

:@l
dense_3/truediv_11RealDivdense_3/Abs_3:y:0dense_3/Pow_3:z:0*
T0*
_output_shapes

:@T
dense_3/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?q
dense_3/add_7AddV2dense_3/truediv_11:z:0dense_3/add_7/y:output:0*
T0*
_output_shapes

:@T
dense_3/Floor_2Floordense_3/add_7:z:0*
T0*
_output_shapes

:@U
dense_3/Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_3/Less_2Lessdense_3/Floor_2:y:0dense_3/Less_2/y:output:0*
T0*
_output_shapes

:@T
dense_3/Sign_2Signdense_3/truediv:z:0*
T0*
_output_shapes

:@z
)dense_3/ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      ^
dense_3/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_3/ones_like_2Fill2dense_3/ones_like_2/Shape/shape_as_tensor:output:0"dense_3/ones_like_2/Const:output:0*
T0*
_output_shapes

:@T
dense_3/mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Au
dense_3/mul_9Muldense_3/ones_like_2:output:0dense_3/mul_9/y:output:0*
T0*
_output_shapes

:@Y
dense_3/truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @x
dense_3/truediv_12RealDivdense_3/mul_9:z:0dense_3/truediv_12/y:output:0*
T0*
_output_shapes

:@
dense_3/SelectV2_2SelectV2dense_3/Less_2:z:0dense_3/Floor_2:y:0dense_3/truediv_12:z:0*
T0*
_output_shapes

:@o
dense_3/mul_10Muldense_3/Sign_2:y:0dense_3/SelectV2_2:output:0*
T0*
_output_shapes

:@g
dense_3/Mul_11Muldense_3/truediv:z:0dense_3/mul_10:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_4Meandense_3/Mul_11:z:0)dense_3/Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(f
dense_3/Mul_12Muldense_3/mul_10:z:0dense_3/mul_10:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_5Meandense_3/Mul_12:z:0)dense_3/Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(T
dense_3/add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3r
dense_3/add_8AddV2dense_3/Mean_5:output:0dense_3/add_8/y:output:0*
T0*
_output_shapes

:r
dense_3/truediv_13RealDivdense_3/Mean_4:output:0dense_3/add_8:z:0*
T0*
_output_shapes

:T
dense_3/add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3q
dense_3/add_9AddV2dense_3/truediv_13:z:0dense_3/add_9/y:output:0*
T0*
_output_shapes

:P
dense_3/Log_3Logdense_3/add_9:z:0*
T0*
_output_shapes

:Y
dense_3/truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_3/truediv_14RealDivdense_3/Log_3:y:0dense_3/truediv_14/y:output:0*
T0*
_output_shapes

:Y
dense_3/Round_3Rounddense_3/truediv_14:z:0*
T0*
_output_shapes

:T
dense_3/Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_3/Pow_4Powdense_3/Pow_4/x:output:0dense_3/Round_3:y:0*
T0*
_output_shapes

:R
dense_3/Abs_4Absdense_3/truediv:z:0*
T0*
_output_shapes

:@l
dense_3/truediv_15RealDivdense_3/Abs_4:y:0dense_3/Pow_4:z:0*
T0*
_output_shapes

:@U
dense_3/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?s
dense_3/add_10AddV2dense_3/truediv_15:z:0dense_3/add_10/y:output:0*
T0*
_output_shapes

:@U
dense_3/Floor_3Floordense_3/add_10:z:0*
T0*
_output_shapes

:@U
dense_3/Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_3/Less_3Lessdense_3/Floor_3:y:0dense_3/Less_3/y:output:0*
T0*
_output_shapes

:@T
dense_3/Sign_3Signdense_3/truediv:z:0*
T0*
_output_shapes

:@z
)dense_3/ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      ^
dense_3/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_3/ones_like_3Fill2dense_3/ones_like_3/Shape/shape_as_tensor:output:0"dense_3/ones_like_3/Const:output:0*
T0*
_output_shapes

:@U
dense_3/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Aw
dense_3/mul_13Muldense_3/ones_like_3:output:0dense_3/mul_13/y:output:0*
T0*
_output_shapes

:@Y
dense_3/truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @y
dense_3/truediv_16RealDivdense_3/mul_13:z:0dense_3/truediv_16/y:output:0*
T0*
_output_shapes

:@
dense_3/SelectV2_3SelectV2dense_3/Less_3:z:0dense_3/Floor_3:y:0dense_3/truediv_16:z:0*
T0*
_output_shapes

:@o
dense_3/mul_14Muldense_3/Sign_3:y:0dense_3/SelectV2_3:output:0*
T0*
_output_shapes

:@g
dense_3/Mul_15Muldense_3/truediv:z:0dense_3/mul_14:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_6Meandense_3/Mul_15:z:0)dense_3/Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(f
dense_3/Mul_16Muldense_3/mul_14:z:0dense_3/mul_14:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_7Meandense_3/Mul_16:z:0)dense_3/Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(U
dense_3/add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3t
dense_3/add_11AddV2dense_3/Mean_7:output:0dense_3/add_11/y:output:0*
T0*
_output_shapes

:s
dense_3/truediv_17RealDivdense_3/Mean_6:output:0dense_3/add_11:z:0*
T0*
_output_shapes

:U
dense_3/add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3s
dense_3/add_12AddV2dense_3/truediv_17:z:0dense_3/add_12/y:output:0*
T0*
_output_shapes

:Q
dense_3/Log_4Logdense_3/add_12:z:0*
T0*
_output_shapes

:Y
dense_3/truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_3/truediv_18RealDivdense_3/Log_4:y:0dense_3/truediv_18/y:output:0*
T0*
_output_shapes

:Y
dense_3/Round_4Rounddense_3/truediv_18:z:0*
T0*
_output_shapes

:T
dense_3/Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_3/Pow_5Powdense_3/Pow_5/x:output:0dense_3/Round_4:y:0*
T0*
_output_shapes

:R
dense_3/Abs_5Absdense_3/truediv:z:0*
T0*
_output_shapes

:@l
dense_3/truediv_19RealDivdense_3/Abs_5:y:0dense_3/Pow_5:z:0*
T0*
_output_shapes

:@U
dense_3/add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?s
dense_3/add_13AddV2dense_3/truediv_19:z:0dense_3/add_13/y:output:0*
T0*
_output_shapes

:@U
dense_3/Floor_4Floordense_3/add_13:z:0*
T0*
_output_shapes

:@U
dense_3/Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@o
dense_3/Less_4Lessdense_3/Floor_4:y:0dense_3/Less_4/y:output:0*
T0*
_output_shapes

:@T
dense_3/Sign_4Signdense_3/truediv:z:0*
T0*
_output_shapes

:@z
)dense_3/ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      ^
dense_3/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_3/ones_like_4Fill2dense_3/ones_like_4/Shape/shape_as_tensor:output:0"dense_3/ones_like_4/Const:output:0*
T0*
_output_shapes

:@U
dense_3/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Aw
dense_3/mul_17Muldense_3/ones_like_4:output:0dense_3/mul_17/y:output:0*
T0*
_output_shapes

:@Y
dense_3/truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @y
dense_3/truediv_20RealDivdense_3/mul_17:z:0dense_3/truediv_20/y:output:0*
T0*
_output_shapes

:@
dense_3/SelectV2_4SelectV2dense_3/Less_4:z:0dense_3/Floor_4:y:0dense_3/truediv_20:z:0*
T0*
_output_shapes

:@o
dense_3/mul_18Muldense_3/Sign_4:y:0dense_3/SelectV2_4:output:0*
T0*
_output_shapes

:@g
dense_3/Mul_19Muldense_3/truediv:z:0dense_3/mul_18:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_8Meandense_3/Mul_19:z:0)dense_3/Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(f
dense_3/Mul_20Muldense_3/mul_18:z:0dense_3/mul_18:z:0*
T0*
_output_shapes

:@j
 dense_3/Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Mean_9Meandense_3/Mul_20:z:0)dense_3/Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(U
dense_3/add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3t
dense_3/add_14AddV2dense_3/Mean_9:output:0dense_3/add_14/y:output:0*
T0*
_output_shapes

:s
dense_3/truediv_21RealDivdense_3/Mean_8:output:0dense_3/add_14:z:0*
T0*
_output_shapes

:U
dense_3/add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3s
dense_3/add_15AddV2dense_3/truediv_21:z:0dense_3/add_15/y:output:0*
T0*
_output_shapes

:Q
dense_3/Log_5Logdense_3/add_15:z:0*
T0*
_output_shapes

:Y
dense_3/truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?x
dense_3/truediv_22RealDivdense_3/Log_5:y:0dense_3/truediv_22/y:output:0*
T0*
_output_shapes

:Y
dense_3/Round_5Rounddense_3/truediv_22:z:0*
T0*
_output_shapes

:T
dense_3/Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dense_3/Pow_6Powdense_3/Pow_6/x:output:0dense_3/Round_5:y:0*
T0*
_output_shapes

:U
dense_3/mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Al
dense_3/mul_21Muldense_3/Pow_6:z:0dense_3/mul_21/y:output:0*
T0*
_output_shapes

:e
dense_3/mul_22Muldense_3/Cast:y:0dense_3/truediv:z:0*
T0*
_output_shapes

:@d
dense_3/mul_23Muldense_3/Cast:y:0dense_3/mul_18:z:0*
T0*
_output_shapes

:@Y
dense_3/truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ay
dense_3/truediv_23RealDivdense_3/mul_23:z:0dense_3/truediv_23/y:output:0*
T0*
_output_shapes

:@j
dense_3/mul_24Muldense_3/mul_21:z:0dense_3/truediv_23:z:0*
T0*
_output_shapes

:@O
dense_3/NegNegdense_3/mul_22:z:0*
T0*
_output_shapes

:@e
dense_3/add_16AddV2dense_3/Neg:y:0dense_3/mul_24:z:0*
T0*
_output_shapes

:@U
dense_3/mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
dense_3/mul_25Muldense_3/mul_25/x:output:0dense_3/add_16:z:0*
T0*
_output_shapes

:@a
dense_3/StopGradientStopGradientdense_3/mul_25:z:0*
T0*
_output_shapes

:@s
dense_3/add_17AddV2dense_3/mul_22:z:0dense_3/StopGradient:output:0*
T0*
_output_shapes

:@q
dense_3/MatMulMatMulre_lu_2/add_2:z:0dense_3/add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
dense_3/Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :Q
dense_3/Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : i
dense_3/Pow_7Powdense_3/Pow_7/x:output:0dense_3/Pow_7/y:output:0*
T0*
_output_shapes
: Y
dense_3/Cast_1Castdense_3/Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: v
dense_3/ReadVariableOp_1ReadVariableOp!dense_3_readvariableop_1_resource*
_output_shapes
:*
dtype0U
dense_3/mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aw
dense_3/mul_26Mul dense_3/ReadVariableOp_1:value:0dense_3/mul_26/y:output:0*
T0*
_output_shapes
:j
dense_3/truediv_24RealDivdense_3/mul_26:z:0dense_3/Cast_1:y:0*
T0*
_output_shapes
:Q
dense_3/Neg_1Negdense_3/truediv_24:z:0*
T0*
_output_shapes
:U
dense_3/Round_6Rounddense_3/truediv_24:z:0*
T0*
_output_shapes
:d
dense_3/add_18AddV2dense_3/Neg_1:y:0dense_3/Round_6:y:0*
T0*
_output_shapes
:_
dense_3/StopGradient_1StopGradientdense_3/add_18:z:0*
T0*
_output_shapes
:u
dense_3/add_19AddV2dense_3/truediv_24:z:0dense_3/StopGradient_1:output:0*
T0*
_output_shapes
:d
dense_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
dense_3/clip_by_value/MinimumMinimumdense_3/add_19:z:0(dense_3/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:\
dense_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á
dense_3/clip_by_valueMaximum!dense_3/clip_by_value/Minimum:z:0 dense_3/clip_by_value/y:output:0*
T0*
_output_shapes
:i
dense_3/mul_27Muldense_3/Cast_1:y:0dense_3/clip_by_value:z:0*
T0*
_output_shapes
:Y
dense_3/truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Au
dense_3/truediv_25RealDivdense_3/mul_27:z:0dense_3/truediv_25/y:output:0*
T0*
_output_shapes
:U
dense_3/mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?m
dense_3/mul_28Muldense_3/mul_28/x:output:0dense_3/truediv_25:z:0*
T0*
_output_shapes
:v
dense_3/ReadVariableOp_2ReadVariableOp!dense_3_readvariableop_1_resource*
_output_shapes
:*
dtype0[
dense_3/Neg_2Neg dense_3/ReadVariableOp_2:value:0*
T0*
_output_shapes
:c
dense_3/add_20AddV2dense_3/Neg_2:y:0dense_3/mul_28:z:0*
T0*
_output_shapes
:U
dense_3/mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
dense_3/mul_29Muldense_3/mul_29/x:output:0dense_3/add_20:z:0*
T0*
_output_shapes
:_
dense_3/StopGradient_2StopGradientdense_3/mul_29:z:0*
T0*
_output_shapes
:v
dense_3/ReadVariableOp_3ReadVariableOp!dense_3_readvariableop_1_resource*
_output_shapes
:*
dtype0
dense_3/add_21AddV2 dense_3/ReadVariableOp_3:value:0dense_3/StopGradient_2:output:0*
T0*
_output_shapes
:z
dense_3/BiasAddBiasAdddense_3/MatMul:product:0dense_3/add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp^dense/ReadVariableOp^dense/ReadVariableOp_1^dense/ReadVariableOp_2^dense/ReadVariableOp_3^dense_1/ReadVariableOp^dense_1/ReadVariableOp_1^dense_1/ReadVariableOp_2^dense_1/ReadVariableOp_3^dense_2/ReadVariableOp^dense_2/ReadVariableOp_1^dense_2/ReadVariableOp_2^dense_2/ReadVariableOp_3^dense_3/ReadVariableOp^dense_3/ReadVariableOp_1^dense_3/ReadVariableOp_2^dense_3/ReadVariableOp_3*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿv: : : : : : : : : : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2,
dense/ReadVariableOpdense/ReadVariableOp20
dense/ReadVariableOp_1dense/ReadVariableOp_120
dense/ReadVariableOp_2dense/ReadVariableOp_220
dense/ReadVariableOp_3dense/ReadVariableOp_320
dense_1/ReadVariableOpdense_1/ReadVariableOp24
dense_1/ReadVariableOp_1dense_1/ReadVariableOp_124
dense_1/ReadVariableOp_2dense_1/ReadVariableOp_224
dense_1/ReadVariableOp_3dense_1/ReadVariableOp_320
dense_2/ReadVariableOpdense_2/ReadVariableOp24
dense_2/ReadVariableOp_1dense_2/ReadVariableOp_124
dense_2/ReadVariableOp_2dense_2/ReadVariableOp_224
dense_2/ReadVariableOp_3dense_2/ReadVariableOp_320
dense_3/ReadVariableOpdense_3/ReadVariableOp24
dense_3/ReadVariableOp_1dense_3/ReadVariableOp_124
dense_3/ReadVariableOp_2dense_3/ReadVariableOp_224
dense_3/ReadVariableOp_3dense_3/ReadVariableOp_3:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_nameinputs
è¨
ÿ
B__inference_dense_3_layer_call_and_return_conditional_losses_76495

inputs)
readvariableop_resource:@'
readvariableop_1_resource:
identity¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0]
truedivRealDivReadVariableOp:value:0Cast:y:0*
T0*
_output_shapes

:@@
AbsAbstruediv:z:0*
T0*
_output_shapes

:@_
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: m
MaxMaxAbs:y:0Max/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Q
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A\
	truediv_1RealDivmul:z:0truediv_1/y:output:0*
T0*
_output_shapes

:J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3T
addAddV2truediv_1:z:0add/y:output:0*
T0*
_output_shapes

:<
LogLogadd:z:0*
T0*
_output_shapes

:P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?\
	truediv_2RealDivLog:y:0truediv_2/y:output:0*
T0*
_output_shapes

:F
RoundRoundtruediv_2:z:0*
T0*
_output_shapes

:L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @R
Pow_1PowPow_1/x:output:0	Round:y:0*
T0*
_output_shapes

:B
Abs_1Abstruediv:z:0*
T0*
_output_shapes

:@S
	truediv_3RealDiv	Abs_1:y:0	Pow_1:z:0*
T0*
_output_shapes

:@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
add_1AddV2truediv_3:z:0add_1/y:output:0*
T0*
_output_shapes

:@B
FloorFloor	add_1:z:0*
T0*
_output_shapes

:@K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@Q
LessLess	Floor:y:0Less/y:output:0*
T0*
_output_shapes

:@B
SignSigntruediv:z:0*
T0*
_output_shapes

:@p
ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
	ones_likeFill(ones_like/Shape/shape_as_tensor:output:0ones_like/Const:output:0*
T0*
_output_shapes

:@L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A[
mul_1Mulones_like:output:0mul_1/y:output:0*
T0*
_output_shapes

:@P
truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
	truediv_4RealDiv	mul_1:z:0truediv_4/y:output:0*
T0*
_output_shapes

:@a
SelectV2SelectV2Less:z:0	Floor:y:0truediv_4:z:0*
T0*
_output_shapes

:@R
mul_2MulSign:y:0SelectV2:output:0*
T0*
_output_shapes

:@M
Mul_3Multruediv:z:0	mul_2:z:0*
T0*
_output_shapes

:@`
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: r
MeanMean	Mul_3:z:0Mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(K
Mul_4Mul	mul_2:z:0	mul_2:z:0*
T0*
_output_shapes

:@b
Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_1Mean	Mul_4:z:0!Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_2AddV2Mean_1:output:0add_2/y:output:0*
T0*
_output_shapes

:W
	truediv_5RealDivMean:output:0	add_2:z:0*
T0*
_output_shapes

:L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3X
add_3AddV2truediv_5:z:0add_3/y:output:0*
T0*
_output_shapes

:@
Log_1Log	add_3:z:0*
T0*
_output_shapes

:P
truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?^
	truediv_6RealDiv	Log_1:y:0truediv_6/y:output:0*
T0*
_output_shapes

:H
Round_1Roundtruediv_6:z:0*
T0*
_output_shapes

:L
Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_2PowPow_2/x:output:0Round_1:y:0*
T0*
_output_shapes

:B
Abs_2Abstruediv:z:0*
T0*
_output_shapes

:@S
	truediv_7RealDiv	Abs_2:y:0	Pow_2:z:0*
T0*
_output_shapes

:@L
add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
add_4AddV2truediv_7:z:0add_4/y:output:0*
T0*
_output_shapes

:@D
Floor_1Floor	add_4:z:0*
T0*
_output_shapes

:@M
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_1LessFloor_1:y:0Less_1/y:output:0*
T0*
_output_shapes

:@D
Sign_1Signtruediv:z:0*
T0*
_output_shapes

:@r
!ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_1Fill*ones_like_1/Shape/shape_as_tensor:output:0ones_like_1/Const:output:0*
T0*
_output_shapes

:@L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A]
mul_5Mulones_like_1:output:0mul_5/y:output:0*
T0*
_output_shapes

:@P
truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
	truediv_8RealDiv	mul_5:z:0truediv_8/y:output:0*
T0*
_output_shapes

:@g

SelectV2_1SelectV2
Less_1:z:0Floor_1:y:0truediv_8:z:0*
T0*
_output_shapes

:@V
mul_6Mul
Sign_1:y:0SelectV2_1:output:0*
T0*
_output_shapes

:@M
Mul_7Multruediv:z:0	mul_6:z:0*
T0*
_output_shapes

:@b
Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_2Mean	Mul_7:z:0!Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(K
Mul_8Mul	mul_6:z:0	mul_6:z:0*
T0*
_output_shapes

:@b
Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_3Mean	Mul_8:z:0!Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_5AddV2Mean_3:output:0add_5/y:output:0*
T0*
_output_shapes

:Y
	truediv_9RealDivMean_2:output:0	add_5:z:0*
T0*
_output_shapes

:L
add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3X
add_6AddV2truediv_9:z:0add_6/y:output:0*
T0*
_output_shapes

:@
Log_2Log	add_6:z:0*
T0*
_output_shapes

:Q
truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_10RealDiv	Log_2:y:0truediv_10/y:output:0*
T0*
_output_shapes

:I
Round_2Roundtruediv_10:z:0*
T0*
_output_shapes

:L
Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_3PowPow_3/x:output:0Round_2:y:0*
T0*
_output_shapes

:B
Abs_3Abstruediv:z:0*
T0*
_output_shapes

:@T

truediv_11RealDiv	Abs_3:y:0	Pow_3:z:0*
T0*
_output_shapes

:@L
add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Y
add_7AddV2truediv_11:z:0add_7/y:output:0*
T0*
_output_shapes

:@D
Floor_2Floor	add_7:z:0*
T0*
_output_shapes

:@M
Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_2LessFloor_2:y:0Less_2/y:output:0*
T0*
_output_shapes

:@D
Sign_2Signtruediv:z:0*
T0*
_output_shapes

:@r
!ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_2Fill*ones_like_2/Shape/shape_as_tensor:output:0ones_like_2/Const:output:0*
T0*
_output_shapes

:@L
mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A]
mul_9Mulones_like_2:output:0mul_9/y:output:0*
T0*
_output_shapes

:@Q
truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @`

truediv_12RealDiv	mul_9:z:0truediv_12/y:output:0*
T0*
_output_shapes

:@h

SelectV2_2SelectV2
Less_2:z:0Floor_2:y:0truediv_12:z:0*
T0*
_output_shapes

:@W
mul_10Mul
Sign_2:y:0SelectV2_2:output:0*
T0*
_output_shapes

:@O
Mul_11Multruediv:z:0
mul_10:z:0*
T0*
_output_shapes

:@b
Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_4Mean
Mul_11:z:0!Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(N
Mul_12Mul
mul_10:z:0
mul_10:z:0*
T0*
_output_shapes

:@b
Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_5Mean
Mul_12:z:0!Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(L
add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_8AddV2Mean_5:output:0add_8/y:output:0*
T0*
_output_shapes

:Z

truediv_13RealDivMean_4:output:0	add_8:z:0*
T0*
_output_shapes

:L
add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Y
add_9AddV2truediv_13:z:0add_9/y:output:0*
T0*
_output_shapes

:@
Log_3Log	add_9:z:0*
T0*
_output_shapes

:Q
truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_14RealDiv	Log_3:y:0truediv_14/y:output:0*
T0*
_output_shapes

:I
Round_3Roundtruediv_14:z:0*
T0*
_output_shapes

:L
Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_4PowPow_4/x:output:0Round_3:y:0*
T0*
_output_shapes

:B
Abs_4Abstruediv:z:0*
T0*
_output_shapes

:@T

truediv_15RealDiv	Abs_4:y:0	Pow_4:z:0*
T0*
_output_shapes

:@M
add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_10AddV2truediv_15:z:0add_10/y:output:0*
T0*
_output_shapes

:@E
Floor_3Floor
add_10:z:0*
T0*
_output_shapes

:@M
Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_3LessFloor_3:y:0Less_3/y:output:0*
T0*
_output_shapes

:@D
Sign_3Signtruediv:z:0*
T0*
_output_shapes

:@r
!ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      V
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_3Fill*ones_like_3/Shape/shape_as_tensor:output:0ones_like_3/Const:output:0*
T0*
_output_shapes

:@M
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A_
mul_13Mulones_like_3:output:0mul_13/y:output:0*
T0*
_output_shapes

:@Q
truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @a

truediv_16RealDiv
mul_13:z:0truediv_16/y:output:0*
T0*
_output_shapes

:@h

SelectV2_3SelectV2
Less_3:z:0Floor_3:y:0truediv_16:z:0*
T0*
_output_shapes

:@W
mul_14Mul
Sign_3:y:0SelectV2_3:output:0*
T0*
_output_shapes

:@O
Mul_15Multruediv:z:0
mul_14:z:0*
T0*
_output_shapes

:@b
Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_6Mean
Mul_15:z:0!Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(N
Mul_16Mul
mul_14:z:0
mul_14:z:0*
T0*
_output_shapes

:@b
Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_7Mean
Mul_16:z:0!Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(M
add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3\
add_11AddV2Mean_7:output:0add_11/y:output:0*
T0*
_output_shapes

:[

truediv_17RealDivMean_6:output:0
add_11:z:0*
T0*
_output_shapes

:M
add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3[
add_12AddV2truediv_17:z:0add_12/y:output:0*
T0*
_output_shapes

:A
Log_4Log
add_12:z:0*
T0*
_output_shapes

:Q
truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_18RealDiv	Log_4:y:0truediv_18/y:output:0*
T0*
_output_shapes

:I
Round_4Roundtruediv_18:z:0*
T0*
_output_shapes

:L
Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_5PowPow_5/x:output:0Round_4:y:0*
T0*
_output_shapes

:B
Abs_5Abstruediv:z:0*
T0*
_output_shapes

:@T

truediv_19RealDiv	Abs_5:y:0	Pow_5:z:0*
T0*
_output_shapes

:@M
add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_13AddV2truediv_19:z:0add_13/y:output:0*
T0*
_output_shapes

:@E
Floor_4Floor
add_13:z:0*
T0*
_output_shapes

:@M
Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_4LessFloor_4:y:0Less_4/y:output:0*
T0*
_output_shapes

:@D
Sign_4Signtruediv:z:0*
T0*
_output_shapes

:@r
!ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      V
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_4Fill*ones_like_4/Shape/shape_as_tensor:output:0ones_like_4/Const:output:0*
T0*
_output_shapes

:@M
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A_
mul_17Mulones_like_4:output:0mul_17/y:output:0*
T0*
_output_shapes

:@Q
truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @a

truediv_20RealDiv
mul_17:z:0truediv_20/y:output:0*
T0*
_output_shapes

:@h

SelectV2_4SelectV2
Less_4:z:0Floor_4:y:0truediv_20:z:0*
T0*
_output_shapes

:@W
mul_18Mul
Sign_4:y:0SelectV2_4:output:0*
T0*
_output_shapes

:@O
Mul_19Multruediv:z:0
mul_18:z:0*
T0*
_output_shapes

:@b
Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_8Mean
Mul_19:z:0!Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(N
Mul_20Mul
mul_18:z:0
mul_18:z:0*
T0*
_output_shapes

:@b
Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_9Mean
Mul_20:z:0!Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(M
add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3\
add_14AddV2Mean_9:output:0add_14/y:output:0*
T0*
_output_shapes

:[

truediv_21RealDivMean_8:output:0
add_14:z:0*
T0*
_output_shapes

:M
add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3[
add_15AddV2truediv_21:z:0add_15/y:output:0*
T0*
_output_shapes

:A
Log_5Log
add_15:z:0*
T0*
_output_shapes

:Q
truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_22RealDiv	Log_5:y:0truediv_22/y:output:0*
T0*
_output_shapes

:I
Round_5Roundtruediv_22:z:0*
T0*
_output_shapes

:L
Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_6PowPow_6/x:output:0Round_5:y:0*
T0*
_output_shapes

:M
mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AT
mul_21Mul	Pow_6:z:0mul_21/y:output:0*
T0*
_output_shapes

:M
mul_22MulCast:y:0truediv:z:0*
T0*
_output_shapes

:@L
mul_23MulCast:y:0
mul_18:z:0*
T0*
_output_shapes

:@Q
truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aa

truediv_23RealDiv
mul_23:z:0truediv_23/y:output:0*
T0*
_output_shapes

:@R
mul_24Mul
mul_21:z:0truediv_23:z:0*
T0*
_output_shapes

:@?
NegNeg
mul_22:z:0*
T0*
_output_shapes

:@M
add_16AddV2Neg:y:0
mul_24:z:0*
T0*
_output_shapes

:@M
mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mul_25Mulmul_25/x:output:0
add_16:z:0*
T0*
_output_shapes

:@Q
StopGradientStopGradient
mul_25:z:0*
T0*
_output_shapes

:@[
add_17AddV2
mul_22:z:0StopGradient:output:0*
T0*
_output_shapes

:@V
MatMulMatMulinputs
add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_7PowPow_7/x:output:0Pow_7/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0M
mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A_
mul_26MulReadVariableOp_1:value:0mul_26/y:output:0*
T0*
_output_shapes
:R

truediv_24RealDiv
mul_26:z:0
Cast_1:y:0*
T0*
_output_shapes
:A
Neg_1Negtruediv_24:z:0*
T0*
_output_shapes
:E
Round_6Roundtruediv_24:z:0*
T0*
_output_shapes
:L
add_18AddV2	Neg_1:y:0Round_6:y:0*
T0*
_output_shapes
:O
StopGradient_1StopGradient
add_18:z:0*
T0*
_output_shapes
:]
add_19AddV2truediv_24:z:0StopGradient_1:output:0*
T0*
_output_shapes
:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@s
clip_by_value/MinimumMinimum
add_19:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ár
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:Q
mul_27Mul
Cast_1:y:0clip_by_value:z:0*
T0*
_output_shapes
:Q
truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A]

truediv_25RealDiv
mul_27:z:0truediv_25/y:output:0*
T0*
_output_shapes
:M
mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mul_28Mulmul_28/x:output:0truediv_25:z:0*
T0*
_output_shapes
:f
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0K
Neg_2NegReadVariableOp_2:value:0*
T0*
_output_shapes
:K
add_20AddV2	Neg_2:y:0
mul_28:z:0*
T0*
_output_shapes
:M
mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Q
mul_29Mulmul_29/x:output:0
add_20:z:0*
T0*
_output_shapes
:O
StopGradient_2StopGradient
mul_29:z:0*
T0*
_output_shapes
:f
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0g
add_21AddV2ReadVariableOp_3:value:0StopGradient_2:output:0*
T0*
_output_shapes
:b
BiasAddBiasAddMatMul:product:0
add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_3:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¾

'__inference_dense_2_layer_call_fn_80037

inputs
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_76213o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ï
b
)__inference_dropout_1_layer_call_fn_79990

inputs
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_76620o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ë
`
'__inference_dropout_layer_call_fn_79611

inputs
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_76659o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

A
%__inference_re_lu_layer_call_fn_79633

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_75687`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¯?
Ö	
H__inference_my_sequential_layer_call_and_return_conditional_losses_76978
input_1
dense_76924:v@
dense_76926:@'
batch_normalization_76929:@'
batch_normalization_76931:@'
batch_normalization_76933:@'
batch_normalization_76935:@
dense_1_76940:@@
dense_1_76942:@)
batch_normalization_1_76945:@)
batch_normalization_1_76947:@)
batch_normalization_1_76949:@)
batch_normalization_1_76951:@
dense_2_76956:@@
dense_2_76958:@)
batch_normalization_2_76961:@)
batch_normalization_2_76963:@)
batch_normalization_2_76965:@)
batch_normalization_2_76967:@
dense_3_76972:@
dense_3_76974:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCallâ
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_76924dense_76926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75649ñ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_76929batch_normalization_76931batch_normalization_76933batch_normalization_76935*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_75225ó
dropout/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_76659Ó
re_lu/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_75687
dense_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0dense_1_76940dense_1_76942*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75931ÿ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_76945batch_normalization_1_76947batch_normalization_1_76949batch_normalization_1_76951*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_75307
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_76620Ù
re_lu_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_75969
dense_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0dense_2_76956dense_2_76958*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_76213ÿ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_76961batch_normalization_2_76963batch_normalization_2_76965batch_normalization_2_76967*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_75389
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_76581Ù
re_lu_2/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_76251
dense_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0dense_3_76972dense_3_76974*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_76495w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿv: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!
_user_specified_name	input_1
ª	
Ì
 __inference__wrapped_model_75154
input_1=
+my_sequential_dense_readvariableop_resource:v@;
-my_sequential_dense_readvariableop_1_resource:@Q
Cmy_sequential_batch_normalization_batchnorm_readvariableop_resource:@U
Gmy_sequential_batch_normalization_batchnorm_mul_readvariableop_resource:@S
Emy_sequential_batch_normalization_batchnorm_readvariableop_1_resource:@S
Emy_sequential_batch_normalization_batchnorm_readvariableop_2_resource:@?
-my_sequential_dense_1_readvariableop_resource:@@=
/my_sequential_dense_1_readvariableop_1_resource:@S
Emy_sequential_batch_normalization_1_batchnorm_readvariableop_resource:@W
Imy_sequential_batch_normalization_1_batchnorm_mul_readvariableop_resource:@U
Gmy_sequential_batch_normalization_1_batchnorm_readvariableop_1_resource:@U
Gmy_sequential_batch_normalization_1_batchnorm_readvariableop_2_resource:@?
-my_sequential_dense_2_readvariableop_resource:@@=
/my_sequential_dense_2_readvariableop_1_resource:@S
Emy_sequential_batch_normalization_2_batchnorm_readvariableop_resource:@W
Imy_sequential_batch_normalization_2_batchnorm_mul_readvariableop_resource:@U
Gmy_sequential_batch_normalization_2_batchnorm_readvariableop_1_resource:@U
Gmy_sequential_batch_normalization_2_batchnorm_readvariableop_2_resource:@?
-my_sequential_dense_3_readvariableop_resource:@=
/my_sequential_dense_3_readvariableop_1_resource:
identity¢:my_sequential/batch_normalization/batchnorm/ReadVariableOp¢<my_sequential/batch_normalization/batchnorm/ReadVariableOp_1¢<my_sequential/batch_normalization/batchnorm/ReadVariableOp_2¢>my_sequential/batch_normalization/batchnorm/mul/ReadVariableOp¢<my_sequential/batch_normalization_1/batchnorm/ReadVariableOp¢>my_sequential/batch_normalization_1/batchnorm/ReadVariableOp_1¢>my_sequential/batch_normalization_1/batchnorm/ReadVariableOp_2¢@my_sequential/batch_normalization_1/batchnorm/mul/ReadVariableOp¢<my_sequential/batch_normalization_2/batchnorm/ReadVariableOp¢>my_sequential/batch_normalization_2/batchnorm/ReadVariableOp_1¢>my_sequential/batch_normalization_2/batchnorm/ReadVariableOp_2¢@my_sequential/batch_normalization_2/batchnorm/mul/ReadVariableOp¢"my_sequential/dense/ReadVariableOp¢$my_sequential/dense/ReadVariableOp_1¢$my_sequential/dense/ReadVariableOp_2¢$my_sequential/dense/ReadVariableOp_3¢$my_sequential/dense_1/ReadVariableOp¢&my_sequential/dense_1/ReadVariableOp_1¢&my_sequential/dense_1/ReadVariableOp_2¢&my_sequential/dense_1/ReadVariableOp_3¢$my_sequential/dense_2/ReadVariableOp¢&my_sequential/dense_2/ReadVariableOp_1¢&my_sequential/dense_2/ReadVariableOp_2¢&my_sequential/dense_2/ReadVariableOp_3¢$my_sequential/dense_3/ReadVariableOp¢&my_sequential/dense_3/ReadVariableOp_1¢&my_sequential/dense_3/ReadVariableOp_2¢&my_sequential/dense_3/ReadVariableOp_3[
my_sequential/dense/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :[
my_sequential/dense/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : 
my_sequential/dense/PowPow"my_sequential/dense/Pow/x:output:0"my_sequential/dense/Pow/y:output:0*
T0*
_output_shapes
: m
my_sequential/dense/CastCastmy_sequential/dense/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: 
"my_sequential/dense/ReadVariableOpReadVariableOp+my_sequential_dense_readvariableop_resource*
_output_shapes

:v@*
dtype0
my_sequential/dense/truedivRealDiv*my_sequential/dense/ReadVariableOp:value:0my_sequential/dense/Cast:y:0*
T0*
_output_shapes

:v@h
my_sequential/dense/AbsAbsmy_sequential/dense/truediv:z:0*
T0*
_output_shapes

:v@s
)my_sequential/dense/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ©
my_sequential/dense/MaxMaxmy_sequential/dense/Abs:y:02my_sequential/dense/Max/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(^
my_sequential/dense/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense/mulMul my_sequential/dense/Max:output:0"my_sequential/dense/mul/y:output:0*
T0*
_output_shapes

:@d
my_sequential/dense/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense/truediv_1RealDivmy_sequential/dense/mul:z:0(my_sequential/dense/truediv_1/y:output:0*
T0*
_output_shapes

:@^
my_sequential/dense/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense/addAddV2!my_sequential/dense/truediv_1:z:0"my_sequential/dense/add/y:output:0*
T0*
_output_shapes

:@d
my_sequential/dense/LogLogmy_sequential/dense/add:z:0*
T0*
_output_shapes

:@d
my_sequential/dense/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?
my_sequential/dense/truediv_2RealDivmy_sequential/dense/Log:y:0(my_sequential/dense/truediv_2/y:output:0*
T0*
_output_shapes

:@n
my_sequential/dense/RoundRound!my_sequential/dense/truediv_2:z:0*
T0*
_output_shapes

:@`
my_sequential/dense/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense/Pow_1Pow$my_sequential/dense/Pow_1/x:output:0my_sequential/dense/Round:y:0*
T0*
_output_shapes

:@j
my_sequential/dense/Abs_1Absmy_sequential/dense/truediv:z:0*
T0*
_output_shapes

:v@
my_sequential/dense/truediv_3RealDivmy_sequential/dense/Abs_1:y:0my_sequential/dense/Pow_1:z:0*
T0*
_output_shapes

:v@`
my_sequential/dense/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense/add_1AddV2!my_sequential/dense/truediv_3:z:0$my_sequential/dense/add_1/y:output:0*
T0*
_output_shapes

:v@j
my_sequential/dense/FloorFloormy_sequential/dense/add_1:z:0*
T0*
_output_shapes

:v@_
my_sequential/dense/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense/LessLessmy_sequential/dense/Floor:y:0#my_sequential/dense/Less/y:output:0*
T0*
_output_shapes

:v@j
my_sequential/dense/SignSignmy_sequential/dense/truediv:z:0*
T0*
_output_shapes

:v@
3my_sequential/dense/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   h
#my_sequential/dense/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?º
my_sequential/dense/ones_likeFill<my_sequential/dense/ones_like/Shape/shape_as_tensor:output:0,my_sequential/dense/ones_like/Const:output:0*
T0*
_output_shapes

:v@`
my_sequential/dense/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense/mul_1Mul&my_sequential/dense/ones_like:output:0$my_sequential/dense/mul_1/y:output:0*
T0*
_output_shapes

:v@d
my_sequential/dense/truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense/truediv_4RealDivmy_sequential/dense/mul_1:z:0(my_sequential/dense/truediv_4/y:output:0*
T0*
_output_shapes

:v@±
my_sequential/dense/SelectV2SelectV2my_sequential/dense/Less:z:0my_sequential/dense/Floor:y:0!my_sequential/dense/truediv_4:z:0*
T0*
_output_shapes

:v@
my_sequential/dense/mul_2Mulmy_sequential/dense/Sign:y:0%my_sequential/dense/SelectV2:output:0*
T0*
_output_shapes

:v@
my_sequential/dense/Mul_3Mulmy_sequential/dense/truediv:z:0my_sequential/dense/mul_2:z:0*
T0*
_output_shapes

:v@t
*my_sequential/dense/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ®
my_sequential/dense/MeanMeanmy_sequential/dense/Mul_3:z:03my_sequential/dense/Mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
my_sequential/dense/Mul_4Mulmy_sequential/dense/mul_2:z:0my_sequential/dense/mul_2:z:0*
T0*
_output_shapes

:v@v
,my_sequential/dense/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ²
my_sequential/dense/Mean_1Meanmy_sequential/dense/Mul_4:z:05my_sequential/dense/Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(`
my_sequential/dense/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense/add_2AddV2#my_sequential/dense/Mean_1:output:0$my_sequential/dense/add_2/y:output:0*
T0*
_output_shapes

:@
my_sequential/dense/truediv_5RealDiv!my_sequential/dense/Mean:output:0my_sequential/dense/add_2:z:0*
T0*
_output_shapes

:@`
my_sequential/dense/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense/add_3AddV2!my_sequential/dense/truediv_5:z:0$my_sequential/dense/add_3/y:output:0*
T0*
_output_shapes

:@h
my_sequential/dense/Log_1Logmy_sequential/dense/add_3:z:0*
T0*
_output_shapes

:@d
my_sequential/dense/truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?
my_sequential/dense/truediv_6RealDivmy_sequential/dense/Log_1:y:0(my_sequential/dense/truediv_6/y:output:0*
T0*
_output_shapes

:@p
my_sequential/dense/Round_1Round!my_sequential/dense/truediv_6:z:0*
T0*
_output_shapes

:@`
my_sequential/dense/Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense/Pow_2Pow$my_sequential/dense/Pow_2/x:output:0my_sequential/dense/Round_1:y:0*
T0*
_output_shapes

:@j
my_sequential/dense/Abs_2Absmy_sequential/dense/truediv:z:0*
T0*
_output_shapes

:v@
my_sequential/dense/truediv_7RealDivmy_sequential/dense/Abs_2:y:0my_sequential/dense/Pow_2:z:0*
T0*
_output_shapes

:v@`
my_sequential/dense/add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense/add_4AddV2!my_sequential/dense/truediv_7:z:0$my_sequential/dense/add_4/y:output:0*
T0*
_output_shapes

:v@l
my_sequential/dense/Floor_1Floormy_sequential/dense/add_4:z:0*
T0*
_output_shapes

:v@a
my_sequential/dense/Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense/Less_1Lessmy_sequential/dense/Floor_1:y:0%my_sequential/dense/Less_1/y:output:0*
T0*
_output_shapes

:v@l
my_sequential/dense/Sign_1Signmy_sequential/dense/truediv:z:0*
T0*
_output_shapes

:v@
5my_sequential/dense/ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   j
%my_sequential/dense/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?À
my_sequential/dense/ones_like_1Fill>my_sequential/dense/ones_like_1/Shape/shape_as_tensor:output:0.my_sequential/dense/ones_like_1/Const:output:0*
T0*
_output_shapes

:v@`
my_sequential/dense/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense/mul_5Mul(my_sequential/dense/ones_like_1:output:0$my_sequential/dense/mul_5/y:output:0*
T0*
_output_shapes

:v@d
my_sequential/dense/truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense/truediv_8RealDivmy_sequential/dense/mul_5:z:0(my_sequential/dense/truediv_8/y:output:0*
T0*
_output_shapes

:v@·
my_sequential/dense/SelectV2_1SelectV2my_sequential/dense/Less_1:z:0my_sequential/dense/Floor_1:y:0!my_sequential/dense/truediv_8:z:0*
T0*
_output_shapes

:v@
my_sequential/dense/mul_6Mulmy_sequential/dense/Sign_1:y:0'my_sequential/dense/SelectV2_1:output:0*
T0*
_output_shapes

:v@
my_sequential/dense/Mul_7Mulmy_sequential/dense/truediv:z:0my_sequential/dense/mul_6:z:0*
T0*
_output_shapes

:v@v
,my_sequential/dense/Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ²
my_sequential/dense/Mean_2Meanmy_sequential/dense/Mul_7:z:05my_sequential/dense/Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
my_sequential/dense/Mul_8Mulmy_sequential/dense/mul_6:z:0my_sequential/dense/mul_6:z:0*
T0*
_output_shapes

:v@v
,my_sequential/dense/Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ²
my_sequential/dense/Mean_3Meanmy_sequential/dense/Mul_8:z:05my_sequential/dense/Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(`
my_sequential/dense/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense/add_5AddV2#my_sequential/dense/Mean_3:output:0$my_sequential/dense/add_5/y:output:0*
T0*
_output_shapes

:@
my_sequential/dense/truediv_9RealDiv#my_sequential/dense/Mean_2:output:0my_sequential/dense/add_5:z:0*
T0*
_output_shapes

:@`
my_sequential/dense/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense/add_6AddV2!my_sequential/dense/truediv_9:z:0$my_sequential/dense/add_6/y:output:0*
T0*
_output_shapes

:@h
my_sequential/dense/Log_2Logmy_sequential/dense/add_6:z:0*
T0*
_output_shapes

:@e
 my_sequential/dense/truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?
my_sequential/dense/truediv_10RealDivmy_sequential/dense/Log_2:y:0)my_sequential/dense/truediv_10/y:output:0*
T0*
_output_shapes

:@q
my_sequential/dense/Round_2Round"my_sequential/dense/truediv_10:z:0*
T0*
_output_shapes

:@`
my_sequential/dense/Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense/Pow_3Pow$my_sequential/dense/Pow_3/x:output:0my_sequential/dense/Round_2:y:0*
T0*
_output_shapes

:@j
my_sequential/dense/Abs_3Absmy_sequential/dense/truediv:z:0*
T0*
_output_shapes

:v@
my_sequential/dense/truediv_11RealDivmy_sequential/dense/Abs_3:y:0my_sequential/dense/Pow_3:z:0*
T0*
_output_shapes

:v@`
my_sequential/dense/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense/add_7AddV2"my_sequential/dense/truediv_11:z:0$my_sequential/dense/add_7/y:output:0*
T0*
_output_shapes

:v@l
my_sequential/dense/Floor_2Floormy_sequential/dense/add_7:z:0*
T0*
_output_shapes

:v@a
my_sequential/dense/Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense/Less_2Lessmy_sequential/dense/Floor_2:y:0%my_sequential/dense/Less_2/y:output:0*
T0*
_output_shapes

:v@l
my_sequential/dense/Sign_2Signmy_sequential/dense/truediv:z:0*
T0*
_output_shapes

:v@
5my_sequential/dense/ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   j
%my_sequential/dense/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?À
my_sequential/dense/ones_like_2Fill>my_sequential/dense/ones_like_2/Shape/shape_as_tensor:output:0.my_sequential/dense/ones_like_2/Const:output:0*
T0*
_output_shapes

:v@`
my_sequential/dense/mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense/mul_9Mul(my_sequential/dense/ones_like_2:output:0$my_sequential/dense/mul_9/y:output:0*
T0*
_output_shapes

:v@e
 my_sequential/dense/truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense/truediv_12RealDivmy_sequential/dense/mul_9:z:0)my_sequential/dense/truediv_12/y:output:0*
T0*
_output_shapes

:v@¸
my_sequential/dense/SelectV2_2SelectV2my_sequential/dense/Less_2:z:0my_sequential/dense/Floor_2:y:0"my_sequential/dense/truediv_12:z:0*
T0*
_output_shapes

:v@
my_sequential/dense/mul_10Mulmy_sequential/dense/Sign_2:y:0'my_sequential/dense/SelectV2_2:output:0*
T0*
_output_shapes

:v@
my_sequential/dense/Mul_11Mulmy_sequential/dense/truediv:z:0my_sequential/dense/mul_10:z:0*
T0*
_output_shapes

:v@v
,my_sequential/dense/Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ³
my_sequential/dense/Mean_4Meanmy_sequential/dense/Mul_11:z:05my_sequential/dense/Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
my_sequential/dense/Mul_12Mulmy_sequential/dense/mul_10:z:0my_sequential/dense/mul_10:z:0*
T0*
_output_shapes

:v@v
,my_sequential/dense/Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ³
my_sequential/dense/Mean_5Meanmy_sequential/dense/Mul_12:z:05my_sequential/dense/Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(`
my_sequential/dense/add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense/add_8AddV2#my_sequential/dense/Mean_5:output:0$my_sequential/dense/add_8/y:output:0*
T0*
_output_shapes

:@
my_sequential/dense/truediv_13RealDiv#my_sequential/dense/Mean_4:output:0my_sequential/dense/add_8:z:0*
T0*
_output_shapes

:@`
my_sequential/dense/add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense/add_9AddV2"my_sequential/dense/truediv_13:z:0$my_sequential/dense/add_9/y:output:0*
T0*
_output_shapes

:@h
my_sequential/dense/Log_3Logmy_sequential/dense/add_9:z:0*
T0*
_output_shapes

:@e
 my_sequential/dense/truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?
my_sequential/dense/truediv_14RealDivmy_sequential/dense/Log_3:y:0)my_sequential/dense/truediv_14/y:output:0*
T0*
_output_shapes

:@q
my_sequential/dense/Round_3Round"my_sequential/dense/truediv_14:z:0*
T0*
_output_shapes

:@`
my_sequential/dense/Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense/Pow_4Pow$my_sequential/dense/Pow_4/x:output:0my_sequential/dense/Round_3:y:0*
T0*
_output_shapes

:@j
my_sequential/dense/Abs_4Absmy_sequential/dense/truediv:z:0*
T0*
_output_shapes

:v@
my_sequential/dense/truediv_15RealDivmy_sequential/dense/Abs_4:y:0my_sequential/dense/Pow_4:z:0*
T0*
_output_shapes

:v@a
my_sequential/dense/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense/add_10AddV2"my_sequential/dense/truediv_15:z:0%my_sequential/dense/add_10/y:output:0*
T0*
_output_shapes

:v@m
my_sequential/dense/Floor_3Floormy_sequential/dense/add_10:z:0*
T0*
_output_shapes

:v@a
my_sequential/dense/Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense/Less_3Lessmy_sequential/dense/Floor_3:y:0%my_sequential/dense/Less_3/y:output:0*
T0*
_output_shapes

:v@l
my_sequential/dense/Sign_3Signmy_sequential/dense/truediv:z:0*
T0*
_output_shapes

:v@
5my_sequential/dense/ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   j
%my_sequential/dense/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?À
my_sequential/dense/ones_like_3Fill>my_sequential/dense/ones_like_3/Shape/shape_as_tensor:output:0.my_sequential/dense/ones_like_3/Const:output:0*
T0*
_output_shapes

:v@a
my_sequential/dense/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense/mul_13Mul(my_sequential/dense/ones_like_3:output:0%my_sequential/dense/mul_13/y:output:0*
T0*
_output_shapes

:v@e
 my_sequential/dense/truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense/truediv_16RealDivmy_sequential/dense/mul_13:z:0)my_sequential/dense/truediv_16/y:output:0*
T0*
_output_shapes

:v@¸
my_sequential/dense/SelectV2_3SelectV2my_sequential/dense/Less_3:z:0my_sequential/dense/Floor_3:y:0"my_sequential/dense/truediv_16:z:0*
T0*
_output_shapes

:v@
my_sequential/dense/mul_14Mulmy_sequential/dense/Sign_3:y:0'my_sequential/dense/SelectV2_3:output:0*
T0*
_output_shapes

:v@
my_sequential/dense/Mul_15Mulmy_sequential/dense/truediv:z:0my_sequential/dense/mul_14:z:0*
T0*
_output_shapes

:v@v
,my_sequential/dense/Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ³
my_sequential/dense/Mean_6Meanmy_sequential/dense/Mul_15:z:05my_sequential/dense/Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
my_sequential/dense/Mul_16Mulmy_sequential/dense/mul_14:z:0my_sequential/dense/mul_14:z:0*
T0*
_output_shapes

:v@v
,my_sequential/dense/Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ³
my_sequential/dense/Mean_7Meanmy_sequential/dense/Mul_16:z:05my_sequential/dense/Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(a
my_sequential/dense/add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense/add_11AddV2#my_sequential/dense/Mean_7:output:0%my_sequential/dense/add_11/y:output:0*
T0*
_output_shapes

:@
my_sequential/dense/truediv_17RealDiv#my_sequential/dense/Mean_6:output:0my_sequential/dense/add_11:z:0*
T0*
_output_shapes

:@a
my_sequential/dense/add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense/add_12AddV2"my_sequential/dense/truediv_17:z:0%my_sequential/dense/add_12/y:output:0*
T0*
_output_shapes

:@i
my_sequential/dense/Log_4Logmy_sequential/dense/add_12:z:0*
T0*
_output_shapes

:@e
 my_sequential/dense/truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?
my_sequential/dense/truediv_18RealDivmy_sequential/dense/Log_4:y:0)my_sequential/dense/truediv_18/y:output:0*
T0*
_output_shapes

:@q
my_sequential/dense/Round_4Round"my_sequential/dense/truediv_18:z:0*
T0*
_output_shapes

:@`
my_sequential/dense/Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense/Pow_5Pow$my_sequential/dense/Pow_5/x:output:0my_sequential/dense/Round_4:y:0*
T0*
_output_shapes

:@j
my_sequential/dense/Abs_5Absmy_sequential/dense/truediv:z:0*
T0*
_output_shapes

:v@
my_sequential/dense/truediv_19RealDivmy_sequential/dense/Abs_5:y:0my_sequential/dense/Pow_5:z:0*
T0*
_output_shapes

:v@a
my_sequential/dense/add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense/add_13AddV2"my_sequential/dense/truediv_19:z:0%my_sequential/dense/add_13/y:output:0*
T0*
_output_shapes

:v@m
my_sequential/dense/Floor_4Floormy_sequential/dense/add_13:z:0*
T0*
_output_shapes

:v@a
my_sequential/dense/Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense/Less_4Lessmy_sequential/dense/Floor_4:y:0%my_sequential/dense/Less_4/y:output:0*
T0*
_output_shapes

:v@l
my_sequential/dense/Sign_4Signmy_sequential/dense/truediv:z:0*
T0*
_output_shapes

:v@
5my_sequential/dense/ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   j
%my_sequential/dense/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?À
my_sequential/dense/ones_like_4Fill>my_sequential/dense/ones_like_4/Shape/shape_as_tensor:output:0.my_sequential/dense/ones_like_4/Const:output:0*
T0*
_output_shapes

:v@a
my_sequential/dense/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense/mul_17Mul(my_sequential/dense/ones_like_4:output:0%my_sequential/dense/mul_17/y:output:0*
T0*
_output_shapes

:v@e
 my_sequential/dense/truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense/truediv_20RealDivmy_sequential/dense/mul_17:z:0)my_sequential/dense/truediv_20/y:output:0*
T0*
_output_shapes

:v@¸
my_sequential/dense/SelectV2_4SelectV2my_sequential/dense/Less_4:z:0my_sequential/dense/Floor_4:y:0"my_sequential/dense/truediv_20:z:0*
T0*
_output_shapes

:v@
my_sequential/dense/mul_18Mulmy_sequential/dense/Sign_4:y:0'my_sequential/dense/SelectV2_4:output:0*
T0*
_output_shapes

:v@
my_sequential/dense/Mul_19Mulmy_sequential/dense/truediv:z:0my_sequential/dense/mul_18:z:0*
T0*
_output_shapes

:v@v
,my_sequential/dense/Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ³
my_sequential/dense/Mean_8Meanmy_sequential/dense/Mul_19:z:05my_sequential/dense/Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
my_sequential/dense/Mul_20Mulmy_sequential/dense/mul_18:z:0my_sequential/dense/mul_18:z:0*
T0*
_output_shapes

:v@v
,my_sequential/dense/Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ³
my_sequential/dense/Mean_9Meanmy_sequential/dense/Mul_20:z:05my_sequential/dense/Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(a
my_sequential/dense/add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense/add_14AddV2#my_sequential/dense/Mean_9:output:0%my_sequential/dense/add_14/y:output:0*
T0*
_output_shapes

:@
my_sequential/dense/truediv_21RealDiv#my_sequential/dense/Mean_8:output:0my_sequential/dense/add_14:z:0*
T0*
_output_shapes

:@a
my_sequential/dense/add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense/add_15AddV2"my_sequential/dense/truediv_21:z:0%my_sequential/dense/add_15/y:output:0*
T0*
_output_shapes

:@i
my_sequential/dense/Log_5Logmy_sequential/dense/add_15:z:0*
T0*
_output_shapes

:@e
 my_sequential/dense/truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?
my_sequential/dense/truediv_22RealDivmy_sequential/dense/Log_5:y:0)my_sequential/dense/truediv_22/y:output:0*
T0*
_output_shapes

:@q
my_sequential/dense/Round_5Round"my_sequential/dense/truediv_22:z:0*
T0*
_output_shapes

:@`
my_sequential/dense/Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense/Pow_6Pow$my_sequential/dense/Pow_6/x:output:0my_sequential/dense/Round_5:y:0*
T0*
_output_shapes

:@a
my_sequential/dense/mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
my_sequential/dense/mul_21Mulmy_sequential/dense/Pow_6:z:0%my_sequential/dense/mul_21/y:output:0*
T0*
_output_shapes

:@
my_sequential/dense/mul_22Mulmy_sequential/dense/Cast:y:0my_sequential/dense/truediv:z:0*
T0*
_output_shapes

:v@
my_sequential/dense/mul_23Mulmy_sequential/dense/Cast:y:0my_sequential/dense/mul_18:z:0*
T0*
_output_shapes

:v@e
 my_sequential/dense/truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
my_sequential/dense/truediv_23RealDivmy_sequential/dense/mul_23:z:0)my_sequential/dense/truediv_23/y:output:0*
T0*
_output_shapes

:v@
my_sequential/dense/mul_24Mulmy_sequential/dense/mul_21:z:0"my_sequential/dense/truediv_23:z:0*
T0*
_output_shapes

:v@g
my_sequential/dense/NegNegmy_sequential/dense/mul_22:z:0*
T0*
_output_shapes

:v@
my_sequential/dense/add_16AddV2my_sequential/dense/Neg:y:0my_sequential/dense/mul_24:z:0*
T0*
_output_shapes

:v@a
my_sequential/dense/mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/dense/mul_25Mul%my_sequential/dense/mul_25/x:output:0my_sequential/dense/add_16:z:0*
T0*
_output_shapes

:v@y
 my_sequential/dense/StopGradientStopGradientmy_sequential/dense/mul_25:z:0*
T0*
_output_shapes

:v@
my_sequential/dense/add_17AddV2my_sequential/dense/mul_22:z:0)my_sequential/dense/StopGradient:output:0*
T0*
_output_shapes

:v@
my_sequential/dense/MatMulMatMulinput_1my_sequential/dense/add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
my_sequential/dense/Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :]
my_sequential/dense/Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : 
my_sequential/dense/Pow_7Pow$my_sequential/dense/Pow_7/x:output:0$my_sequential/dense/Pow_7/y:output:0*
T0*
_output_shapes
: q
my_sequential/dense/Cast_1Castmy_sequential/dense/Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: 
$my_sequential/dense/ReadVariableOp_1ReadVariableOp-my_sequential_dense_readvariableop_1_resource*
_output_shapes
:@*
dtype0a
my_sequential/dense/mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
my_sequential/dense/mul_26Mul,my_sequential/dense/ReadVariableOp_1:value:0%my_sequential/dense/mul_26/y:output:0*
T0*
_output_shapes
:@
my_sequential/dense/truediv_24RealDivmy_sequential/dense/mul_26:z:0my_sequential/dense/Cast_1:y:0*
T0*
_output_shapes
:@i
my_sequential/dense/Neg_1Neg"my_sequential/dense/truediv_24:z:0*
T0*
_output_shapes
:@m
my_sequential/dense/Round_6Round"my_sequential/dense/truediv_24:z:0*
T0*
_output_shapes
:@
my_sequential/dense/add_18AddV2my_sequential/dense/Neg_1:y:0my_sequential/dense/Round_6:y:0*
T0*
_output_shapes
:@w
"my_sequential/dense/StopGradient_1StopGradientmy_sequential/dense/add_18:z:0*
T0*
_output_shapes
:@
my_sequential/dense/add_19AddV2"my_sequential/dense/truediv_24:z:0+my_sequential/dense/StopGradient_1:output:0*
T0*
_output_shapes
:@p
+my_sequential/dense/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@¯
)my_sequential/dense/clip_by_value/MinimumMinimummy_sequential/dense/add_19:z:04my_sequential/dense/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:@h
#my_sequential/dense/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á®
!my_sequential/dense/clip_by_valueMaximum-my_sequential/dense/clip_by_value/Minimum:z:0,my_sequential/dense/clip_by_value/y:output:0*
T0*
_output_shapes
:@
my_sequential/dense/mul_27Mulmy_sequential/dense/Cast_1:y:0%my_sequential/dense/clip_by_value:z:0*
T0*
_output_shapes
:@e
 my_sequential/dense/truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
my_sequential/dense/truediv_25RealDivmy_sequential/dense/mul_27:z:0)my_sequential/dense/truediv_25/y:output:0*
T0*
_output_shapes
:@a
my_sequential/dense/mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/dense/mul_28Mul%my_sequential/dense/mul_28/x:output:0"my_sequential/dense/truediv_25:z:0*
T0*
_output_shapes
:@
$my_sequential/dense/ReadVariableOp_2ReadVariableOp-my_sequential_dense_readvariableop_1_resource*
_output_shapes
:@*
dtype0s
my_sequential/dense/Neg_2Neg,my_sequential/dense/ReadVariableOp_2:value:0*
T0*
_output_shapes
:@
my_sequential/dense/add_20AddV2my_sequential/dense/Neg_2:y:0my_sequential/dense/mul_28:z:0*
T0*
_output_shapes
:@a
my_sequential/dense/mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/dense/mul_29Mul%my_sequential/dense/mul_29/x:output:0my_sequential/dense/add_20:z:0*
T0*
_output_shapes
:@w
"my_sequential/dense/StopGradient_2StopGradientmy_sequential/dense/mul_29:z:0*
T0*
_output_shapes
:@
$my_sequential/dense/ReadVariableOp_3ReadVariableOp-my_sequential_dense_readvariableop_1_resource*
_output_shapes
:@*
dtype0£
my_sequential/dense/add_21AddV2,my_sequential/dense/ReadVariableOp_3:value:0+my_sequential/dense/StopGradient_2:output:0*
T0*
_output_shapes
:@
my_sequential/dense/BiasAddBiasAdd$my_sequential/dense/MatMul:product:0my_sequential/dense/add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
:my_sequential/batch_normalization/batchnorm/ReadVariableOpReadVariableOpCmy_sequential_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0v
1my_sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ý
/my_sequential/batch_normalization/batchnorm/addAddV2Bmy_sequential/batch_normalization/batchnorm/ReadVariableOp:value:0:my_sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:@
1my_sequential/batch_normalization/batchnorm/RsqrtRsqrt3my_sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:@Â
>my_sequential/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpGmy_sequential_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Ú
/my_sequential/batch_normalization/batchnorm/mulMul5my_sequential/batch_normalization/batchnorm/Rsqrt:y:0Fmy_sequential/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@Å
1my_sequential/batch_normalization/batchnorm/mul_1Mul$my_sequential/dense/BiasAdd:output:03my_sequential/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
<my_sequential/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpEmy_sequential_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ø
1my_sequential/batch_normalization/batchnorm/mul_2MulDmy_sequential/batch_normalization/batchnorm/ReadVariableOp_1:value:03my_sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:@¾
<my_sequential/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpEmy_sequential_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0Ø
/my_sequential/batch_normalization/batchnorm/subSubDmy_sequential/batch_normalization/batchnorm/ReadVariableOp_2:value:05my_sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@Ø
1my_sequential/batch_normalization/batchnorm/add_1AddV25my_sequential/batch_normalization/batchnorm/mul_1:z:03my_sequential/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
my_sequential/dropout/IdentityIdentity5my_sequential/batch_normalization/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
my_sequential/re_lu/SignSign'my_sequential/dropout/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
my_sequential/re_lu/AbsAbsmy_sequential/re_lu/Sign:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
my_sequential/re_lu/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/re_lu/subSub"my_sequential/re_lu/sub/x:output:0my_sequential/re_lu/Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
my_sequential/re_lu/addAddV2my_sequential/re_lu/Sign:y:0my_sequential/re_lu/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
my_sequential/re_lu/TanhTanh'my_sequential/dropout/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
my_sequential/re_lu/NegNegmy_sequential/re_lu/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
my_sequential/re_lu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/re_lu/mulMul"my_sequential/re_lu/mul/x:output:0my_sequential/re_lu/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
my_sequential/re_lu/add_1AddV2my_sequential/re_lu/Neg:y:0my_sequential/re_lu/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 my_sequential/re_lu/StopGradientStopGradientmy_sequential/re_lu/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
my_sequential/re_lu/add_2AddV2my_sequential/re_lu/Tanh:y:0)my_sequential/re_lu/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
my_sequential/dense_1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :]
my_sequential/dense_1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : 
my_sequential/dense_1/PowPow$my_sequential/dense_1/Pow/x:output:0$my_sequential/dense_1/Pow/y:output:0*
T0*
_output_shapes
: q
my_sequential/dense_1/CastCastmy_sequential/dense_1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: 
$my_sequential/dense_1/ReadVariableOpReadVariableOp-my_sequential_dense_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
my_sequential/dense_1/truedivRealDiv,my_sequential/dense_1/ReadVariableOp:value:0my_sequential/dense_1/Cast:y:0*
T0*
_output_shapes

:@@l
my_sequential/dense_1/AbsAbs!my_sequential/dense_1/truediv:z:0*
T0*
_output_shapes

:@@u
+my_sequential/dense_1/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¯
my_sequential/dense_1/MaxMaxmy_sequential/dense_1/Abs:y:04my_sequential/dense_1/Max/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(`
my_sequential/dense_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_1/mulMul"my_sequential/dense_1/Max:output:0$my_sequential/dense_1/mul/y:output:0*
T0*
_output_shapes

:@f
!my_sequential/dense_1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense_1/truediv_1RealDivmy_sequential/dense_1/mul:z:0*my_sequential/dense_1/truediv_1/y:output:0*
T0*
_output_shapes

:@`
my_sequential/dense_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_1/addAddV2#my_sequential/dense_1/truediv_1:z:0$my_sequential/dense_1/add/y:output:0*
T0*
_output_shapes

:@h
my_sequential/dense_1/LogLogmy_sequential/dense_1/add:z:0*
T0*
_output_shapes

:@f
!my_sequential/dense_1/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?
my_sequential/dense_1/truediv_2RealDivmy_sequential/dense_1/Log:y:0*my_sequential/dense_1/truediv_2/y:output:0*
T0*
_output_shapes

:@r
my_sequential/dense_1/RoundRound#my_sequential/dense_1/truediv_2:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_1/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_1/Pow_1Pow&my_sequential/dense_1/Pow_1/x:output:0my_sequential/dense_1/Round:y:0*
T0*
_output_shapes

:@n
my_sequential/dense_1/Abs_1Abs!my_sequential/dense_1/truediv:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_1/truediv_3RealDivmy_sequential/dense_1/Abs_1:y:0my_sequential/dense_1/Pow_1:z:0*
T0*
_output_shapes

:@@b
my_sequential/dense_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense_1/add_1AddV2#my_sequential/dense_1/truediv_3:z:0&my_sequential/dense_1/add_1/y:output:0*
T0*
_output_shapes

:@@n
my_sequential/dense_1/FloorFloormy_sequential/dense_1/add_1:z:0*
T0*
_output_shapes

:@@a
my_sequential/dense_1/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense_1/LessLessmy_sequential/dense_1/Floor:y:0%my_sequential/dense_1/Less/y:output:0*
T0*
_output_shapes

:@@n
my_sequential/dense_1/SignSign!my_sequential/dense_1/truediv:z:0*
T0*
_output_shapes

:@@
5my_sequential/dense_1/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   j
%my_sequential/dense_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?À
my_sequential/dense_1/ones_likeFill>my_sequential/dense_1/ones_like/Shape/shape_as_tensor:output:0.my_sequential/dense_1/ones_like/Const:output:0*
T0*
_output_shapes

:@@b
my_sequential/dense_1/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense_1/mul_1Mul(my_sequential/dense_1/ones_like:output:0&my_sequential/dense_1/mul_1/y:output:0*
T0*
_output_shapes

:@@f
!my_sequential/dense_1/truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ 
my_sequential/dense_1/truediv_4RealDivmy_sequential/dense_1/mul_1:z:0*my_sequential/dense_1/truediv_4/y:output:0*
T0*
_output_shapes

:@@¹
my_sequential/dense_1/SelectV2SelectV2my_sequential/dense_1/Less:z:0my_sequential/dense_1/Floor:y:0#my_sequential/dense_1/truediv_4:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_1/mul_2Mulmy_sequential/dense_1/Sign:y:0'my_sequential/dense_1/SelectV2:output:0*
T0*
_output_shapes

:@@
my_sequential/dense_1/Mul_3Mul!my_sequential/dense_1/truediv:z:0my_sequential/dense_1/mul_2:z:0*
T0*
_output_shapes

:@@v
,my_sequential/dense_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ´
my_sequential/dense_1/MeanMeanmy_sequential/dense_1/Mul_3:z:05my_sequential/dense_1/Mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
my_sequential/dense_1/Mul_4Mulmy_sequential/dense_1/mul_2:z:0my_sequential/dense_1/mul_2:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¸
my_sequential/dense_1/Mean_1Meanmy_sequential/dense_1/Mul_4:z:07my_sequential/dense_1/Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(b
my_sequential/dense_1/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_1/add_2AddV2%my_sequential/dense_1/Mean_1:output:0&my_sequential/dense_1/add_2/y:output:0*
T0*
_output_shapes

:@
my_sequential/dense_1/truediv_5RealDiv#my_sequential/dense_1/Mean:output:0my_sequential/dense_1/add_2:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_1/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_1/add_3AddV2#my_sequential/dense_1/truediv_5:z:0&my_sequential/dense_1/add_3/y:output:0*
T0*
_output_shapes

:@l
my_sequential/dense_1/Log_1Logmy_sequential/dense_1/add_3:z:0*
T0*
_output_shapes

:@f
!my_sequential/dense_1/truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1? 
my_sequential/dense_1/truediv_6RealDivmy_sequential/dense_1/Log_1:y:0*my_sequential/dense_1/truediv_6/y:output:0*
T0*
_output_shapes

:@t
my_sequential/dense_1/Round_1Round#my_sequential/dense_1/truediv_6:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_1/Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_1/Pow_2Pow&my_sequential/dense_1/Pow_2/x:output:0!my_sequential/dense_1/Round_1:y:0*
T0*
_output_shapes

:@n
my_sequential/dense_1/Abs_2Abs!my_sequential/dense_1/truediv:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_1/truediv_7RealDivmy_sequential/dense_1/Abs_2:y:0my_sequential/dense_1/Pow_2:z:0*
T0*
_output_shapes

:@@b
my_sequential/dense_1/add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense_1/add_4AddV2#my_sequential/dense_1/truediv_7:z:0&my_sequential/dense_1/add_4/y:output:0*
T0*
_output_shapes

:@@p
my_sequential/dense_1/Floor_1Floormy_sequential/dense_1/add_4:z:0*
T0*
_output_shapes

:@@c
my_sequential/dense_1/Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense_1/Less_1Less!my_sequential/dense_1/Floor_1:y:0'my_sequential/dense_1/Less_1/y:output:0*
T0*
_output_shapes

:@@p
my_sequential/dense_1/Sign_1Sign!my_sequential/dense_1/truediv:z:0*
T0*
_output_shapes

:@@
7my_sequential/dense_1/ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   l
'my_sequential/dense_1/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Æ
!my_sequential/dense_1/ones_like_1Fill@my_sequential/dense_1/ones_like_1/Shape/shape_as_tensor:output:00my_sequential/dense_1/ones_like_1/Const:output:0*
T0*
_output_shapes

:@@b
my_sequential/dense_1/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense_1/mul_5Mul*my_sequential/dense_1/ones_like_1:output:0&my_sequential/dense_1/mul_5/y:output:0*
T0*
_output_shapes

:@@f
!my_sequential/dense_1/truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ 
my_sequential/dense_1/truediv_8RealDivmy_sequential/dense_1/mul_5:z:0*my_sequential/dense_1/truediv_8/y:output:0*
T0*
_output_shapes

:@@¿
 my_sequential/dense_1/SelectV2_1SelectV2 my_sequential/dense_1/Less_1:z:0!my_sequential/dense_1/Floor_1:y:0#my_sequential/dense_1/truediv_8:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_1/mul_6Mul my_sequential/dense_1/Sign_1:y:0)my_sequential/dense_1/SelectV2_1:output:0*
T0*
_output_shapes

:@@
my_sequential/dense_1/Mul_7Mul!my_sequential/dense_1/truediv:z:0my_sequential/dense_1/mul_6:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_1/Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¸
my_sequential/dense_1/Mean_2Meanmy_sequential/dense_1/Mul_7:z:07my_sequential/dense_1/Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
my_sequential/dense_1/Mul_8Mulmy_sequential/dense_1/mul_6:z:0my_sequential/dense_1/mul_6:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_1/Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¸
my_sequential/dense_1/Mean_3Meanmy_sequential/dense_1/Mul_8:z:07my_sequential/dense_1/Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(b
my_sequential/dense_1/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_1/add_5AddV2%my_sequential/dense_1/Mean_3:output:0&my_sequential/dense_1/add_5/y:output:0*
T0*
_output_shapes

:@
my_sequential/dense_1/truediv_9RealDiv%my_sequential/dense_1/Mean_2:output:0my_sequential/dense_1/add_5:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_1/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_1/add_6AddV2#my_sequential/dense_1/truediv_9:z:0&my_sequential/dense_1/add_6/y:output:0*
T0*
_output_shapes

:@l
my_sequential/dense_1/Log_2Logmy_sequential/dense_1/add_6:z:0*
T0*
_output_shapes

:@g
"my_sequential/dense_1/truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?¢
 my_sequential/dense_1/truediv_10RealDivmy_sequential/dense_1/Log_2:y:0+my_sequential/dense_1/truediv_10/y:output:0*
T0*
_output_shapes

:@u
my_sequential/dense_1/Round_2Round$my_sequential/dense_1/truediv_10:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_1/Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_1/Pow_3Pow&my_sequential/dense_1/Pow_3/x:output:0!my_sequential/dense_1/Round_2:y:0*
T0*
_output_shapes

:@n
my_sequential/dense_1/Abs_3Abs!my_sequential/dense_1/truediv:z:0*
T0*
_output_shapes

:@@
 my_sequential/dense_1/truediv_11RealDivmy_sequential/dense_1/Abs_3:y:0my_sequential/dense_1/Pow_3:z:0*
T0*
_output_shapes

:@@b
my_sequential/dense_1/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense_1/add_7AddV2$my_sequential/dense_1/truediv_11:z:0&my_sequential/dense_1/add_7/y:output:0*
T0*
_output_shapes

:@@p
my_sequential/dense_1/Floor_2Floormy_sequential/dense_1/add_7:z:0*
T0*
_output_shapes

:@@c
my_sequential/dense_1/Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense_1/Less_2Less!my_sequential/dense_1/Floor_2:y:0'my_sequential/dense_1/Less_2/y:output:0*
T0*
_output_shapes

:@@p
my_sequential/dense_1/Sign_2Sign!my_sequential/dense_1/truediv:z:0*
T0*
_output_shapes

:@@
7my_sequential/dense_1/ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   l
'my_sequential/dense_1/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Æ
!my_sequential/dense_1/ones_like_2Fill@my_sequential/dense_1/ones_like_2/Shape/shape_as_tensor:output:00my_sequential/dense_1/ones_like_2/Const:output:0*
T0*
_output_shapes

:@@b
my_sequential/dense_1/mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense_1/mul_9Mul*my_sequential/dense_1/ones_like_2:output:0&my_sequential/dense_1/mul_9/y:output:0*
T0*
_output_shapes

:@@g
"my_sequential/dense_1/truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @¢
 my_sequential/dense_1/truediv_12RealDivmy_sequential/dense_1/mul_9:z:0+my_sequential/dense_1/truediv_12/y:output:0*
T0*
_output_shapes

:@@À
 my_sequential/dense_1/SelectV2_2SelectV2 my_sequential/dense_1/Less_2:z:0!my_sequential/dense_1/Floor_2:y:0$my_sequential/dense_1/truediv_12:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_1/mul_10Mul my_sequential/dense_1/Sign_2:y:0)my_sequential/dense_1/SelectV2_2:output:0*
T0*
_output_shapes

:@@
my_sequential/dense_1/Mul_11Mul!my_sequential/dense_1/truediv:z:0 my_sequential/dense_1/mul_10:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_1/Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_1/Mean_4Mean my_sequential/dense_1/Mul_11:z:07my_sequential/dense_1/Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
my_sequential/dense_1/Mul_12Mul my_sequential/dense_1/mul_10:z:0 my_sequential/dense_1/mul_10:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_1/Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_1/Mean_5Mean my_sequential/dense_1/Mul_12:z:07my_sequential/dense_1/Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(b
my_sequential/dense_1/add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_1/add_8AddV2%my_sequential/dense_1/Mean_5:output:0&my_sequential/dense_1/add_8/y:output:0*
T0*
_output_shapes

:@
 my_sequential/dense_1/truediv_13RealDiv%my_sequential/dense_1/Mean_4:output:0my_sequential/dense_1/add_8:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_1/add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_1/add_9AddV2$my_sequential/dense_1/truediv_13:z:0&my_sequential/dense_1/add_9/y:output:0*
T0*
_output_shapes

:@l
my_sequential/dense_1/Log_3Logmy_sequential/dense_1/add_9:z:0*
T0*
_output_shapes

:@g
"my_sequential/dense_1/truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?¢
 my_sequential/dense_1/truediv_14RealDivmy_sequential/dense_1/Log_3:y:0+my_sequential/dense_1/truediv_14/y:output:0*
T0*
_output_shapes

:@u
my_sequential/dense_1/Round_3Round$my_sequential/dense_1/truediv_14:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_1/Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_1/Pow_4Pow&my_sequential/dense_1/Pow_4/x:output:0!my_sequential/dense_1/Round_3:y:0*
T0*
_output_shapes

:@n
my_sequential/dense_1/Abs_4Abs!my_sequential/dense_1/truediv:z:0*
T0*
_output_shapes

:@@
 my_sequential/dense_1/truediv_15RealDivmy_sequential/dense_1/Abs_4:y:0my_sequential/dense_1/Pow_4:z:0*
T0*
_output_shapes

:@@c
my_sequential/dense_1/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense_1/add_10AddV2$my_sequential/dense_1/truediv_15:z:0'my_sequential/dense_1/add_10/y:output:0*
T0*
_output_shapes

:@@q
my_sequential/dense_1/Floor_3Floor my_sequential/dense_1/add_10:z:0*
T0*
_output_shapes

:@@c
my_sequential/dense_1/Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense_1/Less_3Less!my_sequential/dense_1/Floor_3:y:0'my_sequential/dense_1/Less_3/y:output:0*
T0*
_output_shapes

:@@p
my_sequential/dense_1/Sign_3Sign!my_sequential/dense_1/truediv:z:0*
T0*
_output_shapes

:@@
7my_sequential/dense_1/ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   l
'my_sequential/dense_1/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Æ
!my_sequential/dense_1/ones_like_3Fill@my_sequential/dense_1/ones_like_3/Shape/shape_as_tensor:output:00my_sequential/dense_1/ones_like_3/Const:output:0*
T0*
_output_shapes

:@@c
my_sequential/dense_1/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A¡
my_sequential/dense_1/mul_13Mul*my_sequential/dense_1/ones_like_3:output:0'my_sequential/dense_1/mul_13/y:output:0*
T0*
_output_shapes

:@@g
"my_sequential/dense_1/truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @£
 my_sequential/dense_1/truediv_16RealDiv my_sequential/dense_1/mul_13:z:0+my_sequential/dense_1/truediv_16/y:output:0*
T0*
_output_shapes

:@@À
 my_sequential/dense_1/SelectV2_3SelectV2 my_sequential/dense_1/Less_3:z:0!my_sequential/dense_1/Floor_3:y:0$my_sequential/dense_1/truediv_16:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_1/mul_14Mul my_sequential/dense_1/Sign_3:y:0)my_sequential/dense_1/SelectV2_3:output:0*
T0*
_output_shapes

:@@
my_sequential/dense_1/Mul_15Mul!my_sequential/dense_1/truediv:z:0 my_sequential/dense_1/mul_14:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_1/Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_1/Mean_6Mean my_sequential/dense_1/Mul_15:z:07my_sequential/dense_1/Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
my_sequential/dense_1/Mul_16Mul my_sequential/dense_1/mul_14:z:0 my_sequential/dense_1/mul_14:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_1/Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_1/Mean_7Mean my_sequential/dense_1/Mul_16:z:07my_sequential/dense_1/Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(c
my_sequential/dense_1/add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_1/add_11AddV2%my_sequential/dense_1/Mean_7:output:0'my_sequential/dense_1/add_11/y:output:0*
T0*
_output_shapes

:@
 my_sequential/dense_1/truediv_17RealDiv%my_sequential/dense_1/Mean_6:output:0 my_sequential/dense_1/add_11:z:0*
T0*
_output_shapes

:@c
my_sequential/dense_1/add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_1/add_12AddV2$my_sequential/dense_1/truediv_17:z:0'my_sequential/dense_1/add_12/y:output:0*
T0*
_output_shapes

:@m
my_sequential/dense_1/Log_4Log my_sequential/dense_1/add_12:z:0*
T0*
_output_shapes

:@g
"my_sequential/dense_1/truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?¢
 my_sequential/dense_1/truediv_18RealDivmy_sequential/dense_1/Log_4:y:0+my_sequential/dense_1/truediv_18/y:output:0*
T0*
_output_shapes

:@u
my_sequential/dense_1/Round_4Round$my_sequential/dense_1/truediv_18:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_1/Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_1/Pow_5Pow&my_sequential/dense_1/Pow_5/x:output:0!my_sequential/dense_1/Round_4:y:0*
T0*
_output_shapes

:@n
my_sequential/dense_1/Abs_5Abs!my_sequential/dense_1/truediv:z:0*
T0*
_output_shapes

:@@
 my_sequential/dense_1/truediv_19RealDivmy_sequential/dense_1/Abs_5:y:0my_sequential/dense_1/Pow_5:z:0*
T0*
_output_shapes

:@@c
my_sequential/dense_1/add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense_1/add_13AddV2$my_sequential/dense_1/truediv_19:z:0'my_sequential/dense_1/add_13/y:output:0*
T0*
_output_shapes

:@@q
my_sequential/dense_1/Floor_4Floor my_sequential/dense_1/add_13:z:0*
T0*
_output_shapes

:@@c
my_sequential/dense_1/Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense_1/Less_4Less!my_sequential/dense_1/Floor_4:y:0'my_sequential/dense_1/Less_4/y:output:0*
T0*
_output_shapes

:@@p
my_sequential/dense_1/Sign_4Sign!my_sequential/dense_1/truediv:z:0*
T0*
_output_shapes

:@@
7my_sequential/dense_1/ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   l
'my_sequential/dense_1/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Æ
!my_sequential/dense_1/ones_like_4Fill@my_sequential/dense_1/ones_like_4/Shape/shape_as_tensor:output:00my_sequential/dense_1/ones_like_4/Const:output:0*
T0*
_output_shapes

:@@c
my_sequential/dense_1/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A¡
my_sequential/dense_1/mul_17Mul*my_sequential/dense_1/ones_like_4:output:0'my_sequential/dense_1/mul_17/y:output:0*
T0*
_output_shapes

:@@g
"my_sequential/dense_1/truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @£
 my_sequential/dense_1/truediv_20RealDiv my_sequential/dense_1/mul_17:z:0+my_sequential/dense_1/truediv_20/y:output:0*
T0*
_output_shapes

:@@À
 my_sequential/dense_1/SelectV2_4SelectV2 my_sequential/dense_1/Less_4:z:0!my_sequential/dense_1/Floor_4:y:0$my_sequential/dense_1/truediv_20:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_1/mul_18Mul my_sequential/dense_1/Sign_4:y:0)my_sequential/dense_1/SelectV2_4:output:0*
T0*
_output_shapes

:@@
my_sequential/dense_1/Mul_19Mul!my_sequential/dense_1/truediv:z:0 my_sequential/dense_1/mul_18:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_1/Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_1/Mean_8Mean my_sequential/dense_1/Mul_19:z:07my_sequential/dense_1/Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
my_sequential/dense_1/Mul_20Mul my_sequential/dense_1/mul_18:z:0 my_sequential/dense_1/mul_18:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_1/Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_1/Mean_9Mean my_sequential/dense_1/Mul_20:z:07my_sequential/dense_1/Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(c
my_sequential/dense_1/add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_1/add_14AddV2%my_sequential/dense_1/Mean_9:output:0'my_sequential/dense_1/add_14/y:output:0*
T0*
_output_shapes

:@
 my_sequential/dense_1/truediv_21RealDiv%my_sequential/dense_1/Mean_8:output:0 my_sequential/dense_1/add_14:z:0*
T0*
_output_shapes

:@c
my_sequential/dense_1/add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_1/add_15AddV2$my_sequential/dense_1/truediv_21:z:0'my_sequential/dense_1/add_15/y:output:0*
T0*
_output_shapes

:@m
my_sequential/dense_1/Log_5Log my_sequential/dense_1/add_15:z:0*
T0*
_output_shapes

:@g
"my_sequential/dense_1/truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?¢
 my_sequential/dense_1/truediv_22RealDivmy_sequential/dense_1/Log_5:y:0+my_sequential/dense_1/truediv_22/y:output:0*
T0*
_output_shapes

:@u
my_sequential/dense_1/Round_5Round$my_sequential/dense_1/truediv_22:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_1/Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_1/Pow_6Pow&my_sequential/dense_1/Pow_6/x:output:0!my_sequential/dense_1/Round_5:y:0*
T0*
_output_shapes

:@c
my_sequential/dense_1/mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
my_sequential/dense_1/mul_21Mulmy_sequential/dense_1/Pow_6:z:0'my_sequential/dense_1/mul_21/y:output:0*
T0*
_output_shapes

:@
my_sequential/dense_1/mul_22Mulmy_sequential/dense_1/Cast:y:0!my_sequential/dense_1/truediv:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_1/mul_23Mulmy_sequential/dense_1/Cast:y:0 my_sequential/dense_1/mul_18:z:0*
T0*
_output_shapes

:@@g
"my_sequential/dense_1/truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A£
 my_sequential/dense_1/truediv_23RealDiv my_sequential/dense_1/mul_23:z:0+my_sequential/dense_1/truediv_23/y:output:0*
T0*
_output_shapes

:@@
my_sequential/dense_1/mul_24Mul my_sequential/dense_1/mul_21:z:0$my_sequential/dense_1/truediv_23:z:0*
T0*
_output_shapes

:@@k
my_sequential/dense_1/NegNeg my_sequential/dense_1/mul_22:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_1/add_16AddV2my_sequential/dense_1/Neg:y:0 my_sequential/dense_1/mul_24:z:0*
T0*
_output_shapes

:@@c
my_sequential/dense_1/mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/dense_1/mul_25Mul'my_sequential/dense_1/mul_25/x:output:0 my_sequential/dense_1/add_16:z:0*
T0*
_output_shapes

:@@}
"my_sequential/dense_1/StopGradientStopGradient my_sequential/dense_1/mul_25:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_1/add_17AddV2 my_sequential/dense_1/mul_22:z:0+my_sequential/dense_1/StopGradient:output:0*
T0*
_output_shapes

:@@
my_sequential/dense_1/MatMulMatMulmy_sequential/re_lu/add_2:z:0 my_sequential/dense_1/add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
my_sequential/dense_1/Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :_
my_sequential/dense_1/Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : 
my_sequential/dense_1/Pow_7Pow&my_sequential/dense_1/Pow_7/x:output:0&my_sequential/dense_1/Pow_7/y:output:0*
T0*
_output_shapes
: u
my_sequential/dense_1/Cast_1Castmy_sequential/dense_1/Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: 
&my_sequential/dense_1/ReadVariableOp_1ReadVariableOp/my_sequential_dense_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0c
my_sequential/dense_1/mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A¡
my_sequential/dense_1/mul_26Mul.my_sequential/dense_1/ReadVariableOp_1:value:0'my_sequential/dense_1/mul_26/y:output:0*
T0*
_output_shapes
:@
 my_sequential/dense_1/truediv_24RealDiv my_sequential/dense_1/mul_26:z:0 my_sequential/dense_1/Cast_1:y:0*
T0*
_output_shapes
:@m
my_sequential/dense_1/Neg_1Neg$my_sequential/dense_1/truediv_24:z:0*
T0*
_output_shapes
:@q
my_sequential/dense_1/Round_6Round$my_sequential/dense_1/truediv_24:z:0*
T0*
_output_shapes
:@
my_sequential/dense_1/add_18AddV2my_sequential/dense_1/Neg_1:y:0!my_sequential/dense_1/Round_6:y:0*
T0*
_output_shapes
:@{
$my_sequential/dense_1/StopGradient_1StopGradient my_sequential/dense_1/add_18:z:0*
T0*
_output_shapes
:@
my_sequential/dense_1/add_19AddV2$my_sequential/dense_1/truediv_24:z:0-my_sequential/dense_1/StopGradient_1:output:0*
T0*
_output_shapes
:@r
-my_sequential/dense_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@µ
+my_sequential/dense_1/clip_by_value/MinimumMinimum my_sequential/dense_1/add_19:z:06my_sequential/dense_1/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:@j
%my_sequential/dense_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á´
#my_sequential/dense_1/clip_by_valueMaximum/my_sequential/dense_1/clip_by_value/Minimum:z:0.my_sequential/dense_1/clip_by_value/y:output:0*
T0*
_output_shapes
:@
my_sequential/dense_1/mul_27Mul my_sequential/dense_1/Cast_1:y:0'my_sequential/dense_1/clip_by_value:z:0*
T0*
_output_shapes
:@g
"my_sequential/dense_1/truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
 my_sequential/dense_1/truediv_25RealDiv my_sequential/dense_1/mul_27:z:0+my_sequential/dense_1/truediv_25/y:output:0*
T0*
_output_shapes
:@c
my_sequential/dense_1/mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/dense_1/mul_28Mul'my_sequential/dense_1/mul_28/x:output:0$my_sequential/dense_1/truediv_25:z:0*
T0*
_output_shapes
:@
&my_sequential/dense_1/ReadVariableOp_2ReadVariableOp/my_sequential_dense_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0w
my_sequential/dense_1/Neg_2Neg.my_sequential/dense_1/ReadVariableOp_2:value:0*
T0*
_output_shapes
:@
my_sequential/dense_1/add_20AddV2my_sequential/dense_1/Neg_2:y:0 my_sequential/dense_1/mul_28:z:0*
T0*
_output_shapes
:@c
my_sequential/dense_1/mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/dense_1/mul_29Mul'my_sequential/dense_1/mul_29/x:output:0 my_sequential/dense_1/add_20:z:0*
T0*
_output_shapes
:@{
$my_sequential/dense_1/StopGradient_2StopGradient my_sequential/dense_1/mul_29:z:0*
T0*
_output_shapes
:@
&my_sequential/dense_1/ReadVariableOp_3ReadVariableOp/my_sequential_dense_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0©
my_sequential/dense_1/add_21AddV2.my_sequential/dense_1/ReadVariableOp_3:value:0-my_sequential/dense_1/StopGradient_2:output:0*
T0*
_output_shapes
:@¤
my_sequential/dense_1/BiasAddBiasAdd&my_sequential/dense_1/MatMul:product:0 my_sequential/dense_1/add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
<my_sequential/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpEmy_sequential_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0x
3my_sequential/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ã
1my_sequential/batch_normalization_1/batchnorm/addAddV2Dmy_sequential/batch_normalization_1/batchnorm/ReadVariableOp:value:0<my_sequential/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@
3my_sequential/batch_normalization_1/batchnorm/RsqrtRsqrt5my_sequential/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@Æ
@my_sequential/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpImy_sequential_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0à
1my_sequential/batch_normalization_1/batchnorm/mulMul7my_sequential/batch_normalization_1/batchnorm/Rsqrt:y:0Hmy_sequential/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@Ë
3my_sequential/batch_normalization_1/batchnorm/mul_1Mul&my_sequential/dense_1/BiasAdd:output:05my_sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
>my_sequential/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpGmy_sequential_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0Þ
3my_sequential/batch_normalization_1/batchnorm/mul_2MulFmy_sequential/batch_normalization_1/batchnorm/ReadVariableOp_1:value:05my_sequential/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@Â
>my_sequential/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpGmy_sequential_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0Þ
1my_sequential/batch_normalization_1/batchnorm/subSubFmy_sequential/batch_normalization_1/batchnorm/ReadVariableOp_2:value:07my_sequential/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@Þ
3my_sequential/batch_normalization_1/batchnorm/add_1AddV27my_sequential/batch_normalization_1/batchnorm/mul_1:z:05my_sequential/batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 my_sequential/dropout_1/IdentityIdentity7my_sequential/batch_normalization_1/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
my_sequential/re_lu_1/SignSign)my_sequential/dropout_1/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
my_sequential/re_lu_1/AbsAbsmy_sequential/re_lu_1/Sign:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
my_sequential/re_lu_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/re_lu_1/subSub$my_sequential/re_lu_1/sub/x:output:0my_sequential/re_lu_1/Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
my_sequential/re_lu_1/addAddV2my_sequential/re_lu_1/Sign:y:0my_sequential/re_lu_1/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
my_sequential/re_lu_1/TanhTanh)my_sequential/dropout_1/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
my_sequential/re_lu_1/NegNegmy_sequential/re_lu_1/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
my_sequential/re_lu_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/re_lu_1/mulMul$my_sequential/re_lu_1/mul/x:output:0my_sequential/re_lu_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
my_sequential/re_lu_1/add_1AddV2my_sequential/re_lu_1/Neg:y:0my_sequential/re_lu_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"my_sequential/re_lu_1/StopGradientStopGradientmy_sequential/re_lu_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
my_sequential/re_lu_1/add_2AddV2my_sequential/re_lu_1/Tanh:y:0+my_sequential/re_lu_1/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
my_sequential/dense_2/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :]
my_sequential/dense_2/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : 
my_sequential/dense_2/PowPow$my_sequential/dense_2/Pow/x:output:0$my_sequential/dense_2/Pow/y:output:0*
T0*
_output_shapes
: q
my_sequential/dense_2/CastCastmy_sequential/dense_2/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: 
$my_sequential/dense_2/ReadVariableOpReadVariableOp-my_sequential_dense_2_readvariableop_resource*
_output_shapes

:@@*
dtype0
my_sequential/dense_2/truedivRealDiv,my_sequential/dense_2/ReadVariableOp:value:0my_sequential/dense_2/Cast:y:0*
T0*
_output_shapes

:@@l
my_sequential/dense_2/AbsAbs!my_sequential/dense_2/truediv:z:0*
T0*
_output_shapes

:@@u
+my_sequential/dense_2/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¯
my_sequential/dense_2/MaxMaxmy_sequential/dense_2/Abs:y:04my_sequential/dense_2/Max/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(`
my_sequential/dense_2/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_2/mulMul"my_sequential/dense_2/Max:output:0$my_sequential/dense_2/mul/y:output:0*
T0*
_output_shapes

:@f
!my_sequential/dense_2/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense_2/truediv_1RealDivmy_sequential/dense_2/mul:z:0*my_sequential/dense_2/truediv_1/y:output:0*
T0*
_output_shapes

:@`
my_sequential/dense_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_2/addAddV2#my_sequential/dense_2/truediv_1:z:0$my_sequential/dense_2/add/y:output:0*
T0*
_output_shapes

:@h
my_sequential/dense_2/LogLogmy_sequential/dense_2/add:z:0*
T0*
_output_shapes

:@f
!my_sequential/dense_2/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?
my_sequential/dense_2/truediv_2RealDivmy_sequential/dense_2/Log:y:0*my_sequential/dense_2/truediv_2/y:output:0*
T0*
_output_shapes

:@r
my_sequential/dense_2/RoundRound#my_sequential/dense_2/truediv_2:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_2/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_2/Pow_1Pow&my_sequential/dense_2/Pow_1/x:output:0my_sequential/dense_2/Round:y:0*
T0*
_output_shapes

:@n
my_sequential/dense_2/Abs_1Abs!my_sequential/dense_2/truediv:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_2/truediv_3RealDivmy_sequential/dense_2/Abs_1:y:0my_sequential/dense_2/Pow_1:z:0*
T0*
_output_shapes

:@@b
my_sequential/dense_2/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense_2/add_1AddV2#my_sequential/dense_2/truediv_3:z:0&my_sequential/dense_2/add_1/y:output:0*
T0*
_output_shapes

:@@n
my_sequential/dense_2/FloorFloormy_sequential/dense_2/add_1:z:0*
T0*
_output_shapes

:@@a
my_sequential/dense_2/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense_2/LessLessmy_sequential/dense_2/Floor:y:0%my_sequential/dense_2/Less/y:output:0*
T0*
_output_shapes

:@@n
my_sequential/dense_2/SignSign!my_sequential/dense_2/truediv:z:0*
T0*
_output_shapes

:@@
5my_sequential/dense_2/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   j
%my_sequential/dense_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?À
my_sequential/dense_2/ones_likeFill>my_sequential/dense_2/ones_like/Shape/shape_as_tensor:output:0.my_sequential/dense_2/ones_like/Const:output:0*
T0*
_output_shapes

:@@b
my_sequential/dense_2/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense_2/mul_1Mul(my_sequential/dense_2/ones_like:output:0&my_sequential/dense_2/mul_1/y:output:0*
T0*
_output_shapes

:@@f
!my_sequential/dense_2/truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ 
my_sequential/dense_2/truediv_4RealDivmy_sequential/dense_2/mul_1:z:0*my_sequential/dense_2/truediv_4/y:output:0*
T0*
_output_shapes

:@@¹
my_sequential/dense_2/SelectV2SelectV2my_sequential/dense_2/Less:z:0my_sequential/dense_2/Floor:y:0#my_sequential/dense_2/truediv_4:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_2/mul_2Mulmy_sequential/dense_2/Sign:y:0'my_sequential/dense_2/SelectV2:output:0*
T0*
_output_shapes

:@@
my_sequential/dense_2/Mul_3Mul!my_sequential/dense_2/truediv:z:0my_sequential/dense_2/mul_2:z:0*
T0*
_output_shapes

:@@v
,my_sequential/dense_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ´
my_sequential/dense_2/MeanMeanmy_sequential/dense_2/Mul_3:z:05my_sequential/dense_2/Mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
my_sequential/dense_2/Mul_4Mulmy_sequential/dense_2/mul_2:z:0my_sequential/dense_2/mul_2:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_2/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¸
my_sequential/dense_2/Mean_1Meanmy_sequential/dense_2/Mul_4:z:07my_sequential/dense_2/Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(b
my_sequential/dense_2/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_2/add_2AddV2%my_sequential/dense_2/Mean_1:output:0&my_sequential/dense_2/add_2/y:output:0*
T0*
_output_shapes

:@
my_sequential/dense_2/truediv_5RealDiv#my_sequential/dense_2/Mean:output:0my_sequential/dense_2/add_2:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_2/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_2/add_3AddV2#my_sequential/dense_2/truediv_5:z:0&my_sequential/dense_2/add_3/y:output:0*
T0*
_output_shapes

:@l
my_sequential/dense_2/Log_1Logmy_sequential/dense_2/add_3:z:0*
T0*
_output_shapes

:@f
!my_sequential/dense_2/truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1? 
my_sequential/dense_2/truediv_6RealDivmy_sequential/dense_2/Log_1:y:0*my_sequential/dense_2/truediv_6/y:output:0*
T0*
_output_shapes

:@t
my_sequential/dense_2/Round_1Round#my_sequential/dense_2/truediv_6:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_2/Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_2/Pow_2Pow&my_sequential/dense_2/Pow_2/x:output:0!my_sequential/dense_2/Round_1:y:0*
T0*
_output_shapes

:@n
my_sequential/dense_2/Abs_2Abs!my_sequential/dense_2/truediv:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_2/truediv_7RealDivmy_sequential/dense_2/Abs_2:y:0my_sequential/dense_2/Pow_2:z:0*
T0*
_output_shapes

:@@b
my_sequential/dense_2/add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense_2/add_4AddV2#my_sequential/dense_2/truediv_7:z:0&my_sequential/dense_2/add_4/y:output:0*
T0*
_output_shapes

:@@p
my_sequential/dense_2/Floor_1Floormy_sequential/dense_2/add_4:z:0*
T0*
_output_shapes

:@@c
my_sequential/dense_2/Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense_2/Less_1Less!my_sequential/dense_2/Floor_1:y:0'my_sequential/dense_2/Less_1/y:output:0*
T0*
_output_shapes

:@@p
my_sequential/dense_2/Sign_1Sign!my_sequential/dense_2/truediv:z:0*
T0*
_output_shapes

:@@
7my_sequential/dense_2/ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   l
'my_sequential/dense_2/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Æ
!my_sequential/dense_2/ones_like_1Fill@my_sequential/dense_2/ones_like_1/Shape/shape_as_tensor:output:00my_sequential/dense_2/ones_like_1/Const:output:0*
T0*
_output_shapes

:@@b
my_sequential/dense_2/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense_2/mul_5Mul*my_sequential/dense_2/ones_like_1:output:0&my_sequential/dense_2/mul_5/y:output:0*
T0*
_output_shapes

:@@f
!my_sequential/dense_2/truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ 
my_sequential/dense_2/truediv_8RealDivmy_sequential/dense_2/mul_5:z:0*my_sequential/dense_2/truediv_8/y:output:0*
T0*
_output_shapes

:@@¿
 my_sequential/dense_2/SelectV2_1SelectV2 my_sequential/dense_2/Less_1:z:0!my_sequential/dense_2/Floor_1:y:0#my_sequential/dense_2/truediv_8:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_2/mul_6Mul my_sequential/dense_2/Sign_1:y:0)my_sequential/dense_2/SelectV2_1:output:0*
T0*
_output_shapes

:@@
my_sequential/dense_2/Mul_7Mul!my_sequential/dense_2/truediv:z:0my_sequential/dense_2/mul_6:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_2/Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¸
my_sequential/dense_2/Mean_2Meanmy_sequential/dense_2/Mul_7:z:07my_sequential/dense_2/Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
my_sequential/dense_2/Mul_8Mulmy_sequential/dense_2/mul_6:z:0my_sequential/dense_2/mul_6:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_2/Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¸
my_sequential/dense_2/Mean_3Meanmy_sequential/dense_2/Mul_8:z:07my_sequential/dense_2/Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(b
my_sequential/dense_2/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_2/add_5AddV2%my_sequential/dense_2/Mean_3:output:0&my_sequential/dense_2/add_5/y:output:0*
T0*
_output_shapes

:@
my_sequential/dense_2/truediv_9RealDiv%my_sequential/dense_2/Mean_2:output:0my_sequential/dense_2/add_5:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_2/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_2/add_6AddV2#my_sequential/dense_2/truediv_9:z:0&my_sequential/dense_2/add_6/y:output:0*
T0*
_output_shapes

:@l
my_sequential/dense_2/Log_2Logmy_sequential/dense_2/add_6:z:0*
T0*
_output_shapes

:@g
"my_sequential/dense_2/truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?¢
 my_sequential/dense_2/truediv_10RealDivmy_sequential/dense_2/Log_2:y:0+my_sequential/dense_2/truediv_10/y:output:0*
T0*
_output_shapes

:@u
my_sequential/dense_2/Round_2Round$my_sequential/dense_2/truediv_10:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_2/Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_2/Pow_3Pow&my_sequential/dense_2/Pow_3/x:output:0!my_sequential/dense_2/Round_2:y:0*
T0*
_output_shapes

:@n
my_sequential/dense_2/Abs_3Abs!my_sequential/dense_2/truediv:z:0*
T0*
_output_shapes

:@@
 my_sequential/dense_2/truediv_11RealDivmy_sequential/dense_2/Abs_3:y:0my_sequential/dense_2/Pow_3:z:0*
T0*
_output_shapes

:@@b
my_sequential/dense_2/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense_2/add_7AddV2$my_sequential/dense_2/truediv_11:z:0&my_sequential/dense_2/add_7/y:output:0*
T0*
_output_shapes

:@@p
my_sequential/dense_2/Floor_2Floormy_sequential/dense_2/add_7:z:0*
T0*
_output_shapes

:@@c
my_sequential/dense_2/Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense_2/Less_2Less!my_sequential/dense_2/Floor_2:y:0'my_sequential/dense_2/Less_2/y:output:0*
T0*
_output_shapes

:@@p
my_sequential/dense_2/Sign_2Sign!my_sequential/dense_2/truediv:z:0*
T0*
_output_shapes

:@@
7my_sequential/dense_2/ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   l
'my_sequential/dense_2/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Æ
!my_sequential/dense_2/ones_like_2Fill@my_sequential/dense_2/ones_like_2/Shape/shape_as_tensor:output:00my_sequential/dense_2/ones_like_2/Const:output:0*
T0*
_output_shapes

:@@b
my_sequential/dense_2/mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense_2/mul_9Mul*my_sequential/dense_2/ones_like_2:output:0&my_sequential/dense_2/mul_9/y:output:0*
T0*
_output_shapes

:@@g
"my_sequential/dense_2/truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @¢
 my_sequential/dense_2/truediv_12RealDivmy_sequential/dense_2/mul_9:z:0+my_sequential/dense_2/truediv_12/y:output:0*
T0*
_output_shapes

:@@À
 my_sequential/dense_2/SelectV2_2SelectV2 my_sequential/dense_2/Less_2:z:0!my_sequential/dense_2/Floor_2:y:0$my_sequential/dense_2/truediv_12:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_2/mul_10Mul my_sequential/dense_2/Sign_2:y:0)my_sequential/dense_2/SelectV2_2:output:0*
T0*
_output_shapes

:@@
my_sequential/dense_2/Mul_11Mul!my_sequential/dense_2/truediv:z:0 my_sequential/dense_2/mul_10:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_2/Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_2/Mean_4Mean my_sequential/dense_2/Mul_11:z:07my_sequential/dense_2/Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
my_sequential/dense_2/Mul_12Mul my_sequential/dense_2/mul_10:z:0 my_sequential/dense_2/mul_10:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_2/Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_2/Mean_5Mean my_sequential/dense_2/Mul_12:z:07my_sequential/dense_2/Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(b
my_sequential/dense_2/add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_2/add_8AddV2%my_sequential/dense_2/Mean_5:output:0&my_sequential/dense_2/add_8/y:output:0*
T0*
_output_shapes

:@
 my_sequential/dense_2/truediv_13RealDiv%my_sequential/dense_2/Mean_4:output:0my_sequential/dense_2/add_8:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_2/add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_2/add_9AddV2$my_sequential/dense_2/truediv_13:z:0&my_sequential/dense_2/add_9/y:output:0*
T0*
_output_shapes

:@l
my_sequential/dense_2/Log_3Logmy_sequential/dense_2/add_9:z:0*
T0*
_output_shapes

:@g
"my_sequential/dense_2/truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?¢
 my_sequential/dense_2/truediv_14RealDivmy_sequential/dense_2/Log_3:y:0+my_sequential/dense_2/truediv_14/y:output:0*
T0*
_output_shapes

:@u
my_sequential/dense_2/Round_3Round$my_sequential/dense_2/truediv_14:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_2/Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_2/Pow_4Pow&my_sequential/dense_2/Pow_4/x:output:0!my_sequential/dense_2/Round_3:y:0*
T0*
_output_shapes

:@n
my_sequential/dense_2/Abs_4Abs!my_sequential/dense_2/truediv:z:0*
T0*
_output_shapes

:@@
 my_sequential/dense_2/truediv_15RealDivmy_sequential/dense_2/Abs_4:y:0my_sequential/dense_2/Pow_4:z:0*
T0*
_output_shapes

:@@c
my_sequential/dense_2/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense_2/add_10AddV2$my_sequential/dense_2/truediv_15:z:0'my_sequential/dense_2/add_10/y:output:0*
T0*
_output_shapes

:@@q
my_sequential/dense_2/Floor_3Floor my_sequential/dense_2/add_10:z:0*
T0*
_output_shapes

:@@c
my_sequential/dense_2/Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense_2/Less_3Less!my_sequential/dense_2/Floor_3:y:0'my_sequential/dense_2/Less_3/y:output:0*
T0*
_output_shapes

:@@p
my_sequential/dense_2/Sign_3Sign!my_sequential/dense_2/truediv:z:0*
T0*
_output_shapes

:@@
7my_sequential/dense_2/ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   l
'my_sequential/dense_2/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Æ
!my_sequential/dense_2/ones_like_3Fill@my_sequential/dense_2/ones_like_3/Shape/shape_as_tensor:output:00my_sequential/dense_2/ones_like_3/Const:output:0*
T0*
_output_shapes

:@@c
my_sequential/dense_2/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A¡
my_sequential/dense_2/mul_13Mul*my_sequential/dense_2/ones_like_3:output:0'my_sequential/dense_2/mul_13/y:output:0*
T0*
_output_shapes

:@@g
"my_sequential/dense_2/truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @£
 my_sequential/dense_2/truediv_16RealDiv my_sequential/dense_2/mul_13:z:0+my_sequential/dense_2/truediv_16/y:output:0*
T0*
_output_shapes

:@@À
 my_sequential/dense_2/SelectV2_3SelectV2 my_sequential/dense_2/Less_3:z:0!my_sequential/dense_2/Floor_3:y:0$my_sequential/dense_2/truediv_16:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_2/mul_14Mul my_sequential/dense_2/Sign_3:y:0)my_sequential/dense_2/SelectV2_3:output:0*
T0*
_output_shapes

:@@
my_sequential/dense_2/Mul_15Mul!my_sequential/dense_2/truediv:z:0 my_sequential/dense_2/mul_14:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_2/Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_2/Mean_6Mean my_sequential/dense_2/Mul_15:z:07my_sequential/dense_2/Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
my_sequential/dense_2/Mul_16Mul my_sequential/dense_2/mul_14:z:0 my_sequential/dense_2/mul_14:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_2/Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_2/Mean_7Mean my_sequential/dense_2/Mul_16:z:07my_sequential/dense_2/Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(c
my_sequential/dense_2/add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_2/add_11AddV2%my_sequential/dense_2/Mean_7:output:0'my_sequential/dense_2/add_11/y:output:0*
T0*
_output_shapes

:@
 my_sequential/dense_2/truediv_17RealDiv%my_sequential/dense_2/Mean_6:output:0 my_sequential/dense_2/add_11:z:0*
T0*
_output_shapes

:@c
my_sequential/dense_2/add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_2/add_12AddV2$my_sequential/dense_2/truediv_17:z:0'my_sequential/dense_2/add_12/y:output:0*
T0*
_output_shapes

:@m
my_sequential/dense_2/Log_4Log my_sequential/dense_2/add_12:z:0*
T0*
_output_shapes

:@g
"my_sequential/dense_2/truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?¢
 my_sequential/dense_2/truediv_18RealDivmy_sequential/dense_2/Log_4:y:0+my_sequential/dense_2/truediv_18/y:output:0*
T0*
_output_shapes

:@u
my_sequential/dense_2/Round_4Round$my_sequential/dense_2/truediv_18:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_2/Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_2/Pow_5Pow&my_sequential/dense_2/Pow_5/x:output:0!my_sequential/dense_2/Round_4:y:0*
T0*
_output_shapes

:@n
my_sequential/dense_2/Abs_5Abs!my_sequential/dense_2/truediv:z:0*
T0*
_output_shapes

:@@
 my_sequential/dense_2/truediv_19RealDivmy_sequential/dense_2/Abs_5:y:0my_sequential/dense_2/Pow_5:z:0*
T0*
_output_shapes

:@@c
my_sequential/dense_2/add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense_2/add_13AddV2$my_sequential/dense_2/truediv_19:z:0'my_sequential/dense_2/add_13/y:output:0*
T0*
_output_shapes

:@@q
my_sequential/dense_2/Floor_4Floor my_sequential/dense_2/add_13:z:0*
T0*
_output_shapes

:@@c
my_sequential/dense_2/Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense_2/Less_4Less!my_sequential/dense_2/Floor_4:y:0'my_sequential/dense_2/Less_4/y:output:0*
T0*
_output_shapes

:@@p
my_sequential/dense_2/Sign_4Sign!my_sequential/dense_2/truediv:z:0*
T0*
_output_shapes

:@@
7my_sequential/dense_2/ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   l
'my_sequential/dense_2/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Æ
!my_sequential/dense_2/ones_like_4Fill@my_sequential/dense_2/ones_like_4/Shape/shape_as_tensor:output:00my_sequential/dense_2/ones_like_4/Const:output:0*
T0*
_output_shapes

:@@c
my_sequential/dense_2/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A¡
my_sequential/dense_2/mul_17Mul*my_sequential/dense_2/ones_like_4:output:0'my_sequential/dense_2/mul_17/y:output:0*
T0*
_output_shapes

:@@g
"my_sequential/dense_2/truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @£
 my_sequential/dense_2/truediv_20RealDiv my_sequential/dense_2/mul_17:z:0+my_sequential/dense_2/truediv_20/y:output:0*
T0*
_output_shapes

:@@À
 my_sequential/dense_2/SelectV2_4SelectV2 my_sequential/dense_2/Less_4:z:0!my_sequential/dense_2/Floor_4:y:0$my_sequential/dense_2/truediv_20:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_2/mul_18Mul my_sequential/dense_2/Sign_4:y:0)my_sequential/dense_2/SelectV2_4:output:0*
T0*
_output_shapes

:@@
my_sequential/dense_2/Mul_19Mul!my_sequential/dense_2/truediv:z:0 my_sequential/dense_2/mul_18:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_2/Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_2/Mean_8Mean my_sequential/dense_2/Mul_19:z:07my_sequential/dense_2/Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(
my_sequential/dense_2/Mul_20Mul my_sequential/dense_2/mul_18:z:0 my_sequential/dense_2/mul_18:z:0*
T0*
_output_shapes

:@@x
.my_sequential/dense_2/Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_2/Mean_9Mean my_sequential/dense_2/Mul_20:z:07my_sequential/dense_2/Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(c
my_sequential/dense_2/add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_2/add_14AddV2%my_sequential/dense_2/Mean_9:output:0'my_sequential/dense_2/add_14/y:output:0*
T0*
_output_shapes

:@
 my_sequential/dense_2/truediv_21RealDiv%my_sequential/dense_2/Mean_8:output:0 my_sequential/dense_2/add_14:z:0*
T0*
_output_shapes

:@c
my_sequential/dense_2/add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_2/add_15AddV2$my_sequential/dense_2/truediv_21:z:0'my_sequential/dense_2/add_15/y:output:0*
T0*
_output_shapes

:@m
my_sequential/dense_2/Log_5Log my_sequential/dense_2/add_15:z:0*
T0*
_output_shapes

:@g
"my_sequential/dense_2/truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?¢
 my_sequential/dense_2/truediv_22RealDivmy_sequential/dense_2/Log_5:y:0+my_sequential/dense_2/truediv_22/y:output:0*
T0*
_output_shapes

:@u
my_sequential/dense_2/Round_5Round$my_sequential/dense_2/truediv_22:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_2/Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_2/Pow_6Pow&my_sequential/dense_2/Pow_6/x:output:0!my_sequential/dense_2/Round_5:y:0*
T0*
_output_shapes

:@c
my_sequential/dense_2/mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
my_sequential/dense_2/mul_21Mulmy_sequential/dense_2/Pow_6:z:0'my_sequential/dense_2/mul_21/y:output:0*
T0*
_output_shapes

:@
my_sequential/dense_2/mul_22Mulmy_sequential/dense_2/Cast:y:0!my_sequential/dense_2/truediv:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_2/mul_23Mulmy_sequential/dense_2/Cast:y:0 my_sequential/dense_2/mul_18:z:0*
T0*
_output_shapes

:@@g
"my_sequential/dense_2/truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A£
 my_sequential/dense_2/truediv_23RealDiv my_sequential/dense_2/mul_23:z:0+my_sequential/dense_2/truediv_23/y:output:0*
T0*
_output_shapes

:@@
my_sequential/dense_2/mul_24Mul my_sequential/dense_2/mul_21:z:0$my_sequential/dense_2/truediv_23:z:0*
T0*
_output_shapes

:@@k
my_sequential/dense_2/NegNeg my_sequential/dense_2/mul_22:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_2/add_16AddV2my_sequential/dense_2/Neg:y:0 my_sequential/dense_2/mul_24:z:0*
T0*
_output_shapes

:@@c
my_sequential/dense_2/mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/dense_2/mul_25Mul'my_sequential/dense_2/mul_25/x:output:0 my_sequential/dense_2/add_16:z:0*
T0*
_output_shapes

:@@}
"my_sequential/dense_2/StopGradientStopGradient my_sequential/dense_2/mul_25:z:0*
T0*
_output_shapes

:@@
my_sequential/dense_2/add_17AddV2 my_sequential/dense_2/mul_22:z:0+my_sequential/dense_2/StopGradient:output:0*
T0*
_output_shapes

:@@
my_sequential/dense_2/MatMulMatMulmy_sequential/re_lu_1/add_2:z:0 my_sequential/dense_2/add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
my_sequential/dense_2/Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :_
my_sequential/dense_2/Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : 
my_sequential/dense_2/Pow_7Pow&my_sequential/dense_2/Pow_7/x:output:0&my_sequential/dense_2/Pow_7/y:output:0*
T0*
_output_shapes
: u
my_sequential/dense_2/Cast_1Castmy_sequential/dense_2/Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: 
&my_sequential/dense_2/ReadVariableOp_1ReadVariableOp/my_sequential_dense_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0c
my_sequential/dense_2/mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A¡
my_sequential/dense_2/mul_26Mul.my_sequential/dense_2/ReadVariableOp_1:value:0'my_sequential/dense_2/mul_26/y:output:0*
T0*
_output_shapes
:@
 my_sequential/dense_2/truediv_24RealDiv my_sequential/dense_2/mul_26:z:0 my_sequential/dense_2/Cast_1:y:0*
T0*
_output_shapes
:@m
my_sequential/dense_2/Neg_1Neg$my_sequential/dense_2/truediv_24:z:0*
T0*
_output_shapes
:@q
my_sequential/dense_2/Round_6Round$my_sequential/dense_2/truediv_24:z:0*
T0*
_output_shapes
:@
my_sequential/dense_2/add_18AddV2my_sequential/dense_2/Neg_1:y:0!my_sequential/dense_2/Round_6:y:0*
T0*
_output_shapes
:@{
$my_sequential/dense_2/StopGradient_1StopGradient my_sequential/dense_2/add_18:z:0*
T0*
_output_shapes
:@
my_sequential/dense_2/add_19AddV2$my_sequential/dense_2/truediv_24:z:0-my_sequential/dense_2/StopGradient_1:output:0*
T0*
_output_shapes
:@r
-my_sequential/dense_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@µ
+my_sequential/dense_2/clip_by_value/MinimumMinimum my_sequential/dense_2/add_19:z:06my_sequential/dense_2/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:@j
%my_sequential/dense_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á´
#my_sequential/dense_2/clip_by_valueMaximum/my_sequential/dense_2/clip_by_value/Minimum:z:0.my_sequential/dense_2/clip_by_value/y:output:0*
T0*
_output_shapes
:@
my_sequential/dense_2/mul_27Mul my_sequential/dense_2/Cast_1:y:0'my_sequential/dense_2/clip_by_value:z:0*
T0*
_output_shapes
:@g
"my_sequential/dense_2/truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
 my_sequential/dense_2/truediv_25RealDiv my_sequential/dense_2/mul_27:z:0+my_sequential/dense_2/truediv_25/y:output:0*
T0*
_output_shapes
:@c
my_sequential/dense_2/mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/dense_2/mul_28Mul'my_sequential/dense_2/mul_28/x:output:0$my_sequential/dense_2/truediv_25:z:0*
T0*
_output_shapes
:@
&my_sequential/dense_2/ReadVariableOp_2ReadVariableOp/my_sequential_dense_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0w
my_sequential/dense_2/Neg_2Neg.my_sequential/dense_2/ReadVariableOp_2:value:0*
T0*
_output_shapes
:@
my_sequential/dense_2/add_20AddV2my_sequential/dense_2/Neg_2:y:0 my_sequential/dense_2/mul_28:z:0*
T0*
_output_shapes
:@c
my_sequential/dense_2/mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/dense_2/mul_29Mul'my_sequential/dense_2/mul_29/x:output:0 my_sequential/dense_2/add_20:z:0*
T0*
_output_shapes
:@{
$my_sequential/dense_2/StopGradient_2StopGradient my_sequential/dense_2/mul_29:z:0*
T0*
_output_shapes
:@
&my_sequential/dense_2/ReadVariableOp_3ReadVariableOp/my_sequential_dense_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0©
my_sequential/dense_2/add_21AddV2.my_sequential/dense_2/ReadVariableOp_3:value:0-my_sequential/dense_2/StopGradient_2:output:0*
T0*
_output_shapes
:@¤
my_sequential/dense_2/BiasAddBiasAdd&my_sequential/dense_2/MatMul:product:0 my_sequential/dense_2/add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
<my_sequential/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpEmy_sequential_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0x
3my_sequential/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ã
1my_sequential/batch_normalization_2/batchnorm/addAddV2Dmy_sequential/batch_normalization_2/batchnorm/ReadVariableOp:value:0<my_sequential/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@
3my_sequential/batch_normalization_2/batchnorm/RsqrtRsqrt5my_sequential/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@Æ
@my_sequential/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpImy_sequential_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0à
1my_sequential/batch_normalization_2/batchnorm/mulMul7my_sequential/batch_normalization_2/batchnorm/Rsqrt:y:0Hmy_sequential/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@Ë
3my_sequential/batch_normalization_2/batchnorm/mul_1Mul&my_sequential/dense_2/BiasAdd:output:05my_sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Â
>my_sequential/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpGmy_sequential_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0Þ
3my_sequential/batch_normalization_2/batchnorm/mul_2MulFmy_sequential/batch_normalization_2/batchnorm/ReadVariableOp_1:value:05my_sequential/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@Â
>my_sequential/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpGmy_sequential_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0Þ
1my_sequential/batch_normalization_2/batchnorm/subSubFmy_sequential/batch_normalization_2/batchnorm/ReadVariableOp_2:value:07my_sequential/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@Þ
3my_sequential/batch_normalization_2/batchnorm/add_1AddV27my_sequential/batch_normalization_2/batchnorm/mul_1:z:05my_sequential/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 my_sequential/dropout_2/IdentityIdentity7my_sequential/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
my_sequential/re_lu_2/SignSign)my_sequential/dropout_2/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
my_sequential/re_lu_2/AbsAbsmy_sequential/re_lu_2/Sign:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
my_sequential/re_lu_2/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/re_lu_2/subSub$my_sequential/re_lu_2/sub/x:output:0my_sequential/re_lu_2/Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
my_sequential/re_lu_2/addAddV2my_sequential/re_lu_2/Sign:y:0my_sequential/re_lu_2/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
my_sequential/re_lu_2/TanhTanh)my_sequential/dropout_2/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
my_sequential/re_lu_2/NegNegmy_sequential/re_lu_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
my_sequential/re_lu_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/re_lu_2/mulMul$my_sequential/re_lu_2/mul/x:output:0my_sequential/re_lu_2/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
my_sequential/re_lu_2/add_1AddV2my_sequential/re_lu_2/Neg:y:0my_sequential/re_lu_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"my_sequential/re_lu_2/StopGradientStopGradientmy_sequential/re_lu_2/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@£
my_sequential/re_lu_2/add_2AddV2my_sequential/re_lu_2/Tanh:y:0+my_sequential/re_lu_2/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
my_sequential/dense_3/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :]
my_sequential/dense_3/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : 
my_sequential/dense_3/PowPow$my_sequential/dense_3/Pow/x:output:0$my_sequential/dense_3/Pow/y:output:0*
T0*
_output_shapes
: q
my_sequential/dense_3/CastCastmy_sequential/dense_3/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: 
$my_sequential/dense_3/ReadVariableOpReadVariableOp-my_sequential_dense_3_readvariableop_resource*
_output_shapes

:@*
dtype0
my_sequential/dense_3/truedivRealDiv,my_sequential/dense_3/ReadVariableOp:value:0my_sequential/dense_3/Cast:y:0*
T0*
_output_shapes

:@l
my_sequential/dense_3/AbsAbs!my_sequential/dense_3/truediv:z:0*
T0*
_output_shapes

:@u
+my_sequential/dense_3/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¯
my_sequential/dense_3/MaxMaxmy_sequential/dense_3/Abs:y:04my_sequential/dense_3/Max/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(`
my_sequential/dense_3/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_3/mulMul"my_sequential/dense_3/Max:output:0$my_sequential/dense_3/mul/y:output:0*
T0*
_output_shapes

:f
!my_sequential/dense_3/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense_3/truediv_1RealDivmy_sequential/dense_3/mul:z:0*my_sequential/dense_3/truediv_1/y:output:0*
T0*
_output_shapes

:`
my_sequential/dense_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_3/addAddV2#my_sequential/dense_3/truediv_1:z:0$my_sequential/dense_3/add/y:output:0*
T0*
_output_shapes

:h
my_sequential/dense_3/LogLogmy_sequential/dense_3/add:z:0*
T0*
_output_shapes

:f
!my_sequential/dense_3/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?
my_sequential/dense_3/truediv_2RealDivmy_sequential/dense_3/Log:y:0*my_sequential/dense_3/truediv_2/y:output:0*
T0*
_output_shapes

:r
my_sequential/dense_3/RoundRound#my_sequential/dense_3/truediv_2:z:0*
T0*
_output_shapes

:b
my_sequential/dense_3/Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_3/Pow_1Pow&my_sequential/dense_3/Pow_1/x:output:0my_sequential/dense_3/Round:y:0*
T0*
_output_shapes

:n
my_sequential/dense_3/Abs_1Abs!my_sequential/dense_3/truediv:z:0*
T0*
_output_shapes

:@
my_sequential/dense_3/truediv_3RealDivmy_sequential/dense_3/Abs_1:y:0my_sequential/dense_3/Pow_1:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_3/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense_3/add_1AddV2#my_sequential/dense_3/truediv_3:z:0&my_sequential/dense_3/add_1/y:output:0*
T0*
_output_shapes

:@n
my_sequential/dense_3/FloorFloormy_sequential/dense_3/add_1:z:0*
T0*
_output_shapes

:@a
my_sequential/dense_3/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense_3/LessLessmy_sequential/dense_3/Floor:y:0%my_sequential/dense_3/Less/y:output:0*
T0*
_output_shapes

:@n
my_sequential/dense_3/SignSign!my_sequential/dense_3/truediv:z:0*
T0*
_output_shapes

:@
5my_sequential/dense_3/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      j
%my_sequential/dense_3/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?À
my_sequential/dense_3/ones_likeFill>my_sequential/dense_3/ones_like/Shape/shape_as_tensor:output:0.my_sequential/dense_3/ones_like/Const:output:0*
T0*
_output_shapes

:@b
my_sequential/dense_3/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense_3/mul_1Mul(my_sequential/dense_3/ones_like:output:0&my_sequential/dense_3/mul_1/y:output:0*
T0*
_output_shapes

:@f
!my_sequential/dense_3/truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ 
my_sequential/dense_3/truediv_4RealDivmy_sequential/dense_3/mul_1:z:0*my_sequential/dense_3/truediv_4/y:output:0*
T0*
_output_shapes

:@¹
my_sequential/dense_3/SelectV2SelectV2my_sequential/dense_3/Less:z:0my_sequential/dense_3/Floor:y:0#my_sequential/dense_3/truediv_4:z:0*
T0*
_output_shapes

:@
my_sequential/dense_3/mul_2Mulmy_sequential/dense_3/Sign:y:0'my_sequential/dense_3/SelectV2:output:0*
T0*
_output_shapes

:@
my_sequential/dense_3/Mul_3Mul!my_sequential/dense_3/truediv:z:0my_sequential/dense_3/mul_2:z:0*
T0*
_output_shapes

:@v
,my_sequential/dense_3/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ´
my_sequential/dense_3/MeanMeanmy_sequential/dense_3/Mul_3:z:05my_sequential/dense_3/Mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
my_sequential/dense_3/Mul_4Mulmy_sequential/dense_3/mul_2:z:0my_sequential/dense_3/mul_2:z:0*
T0*
_output_shapes

:@x
.my_sequential/dense_3/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¸
my_sequential/dense_3/Mean_1Meanmy_sequential/dense_3/Mul_4:z:07my_sequential/dense_3/Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(b
my_sequential/dense_3/add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_3/add_2AddV2%my_sequential/dense_3/Mean_1:output:0&my_sequential/dense_3/add_2/y:output:0*
T0*
_output_shapes

:
my_sequential/dense_3/truediv_5RealDiv#my_sequential/dense_3/Mean:output:0my_sequential/dense_3/add_2:z:0*
T0*
_output_shapes

:b
my_sequential/dense_3/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_3/add_3AddV2#my_sequential/dense_3/truediv_5:z:0&my_sequential/dense_3/add_3/y:output:0*
T0*
_output_shapes

:l
my_sequential/dense_3/Log_1Logmy_sequential/dense_3/add_3:z:0*
T0*
_output_shapes

:f
!my_sequential/dense_3/truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1? 
my_sequential/dense_3/truediv_6RealDivmy_sequential/dense_3/Log_1:y:0*my_sequential/dense_3/truediv_6/y:output:0*
T0*
_output_shapes

:t
my_sequential/dense_3/Round_1Round#my_sequential/dense_3/truediv_6:z:0*
T0*
_output_shapes

:b
my_sequential/dense_3/Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_3/Pow_2Pow&my_sequential/dense_3/Pow_2/x:output:0!my_sequential/dense_3/Round_1:y:0*
T0*
_output_shapes

:n
my_sequential/dense_3/Abs_2Abs!my_sequential/dense_3/truediv:z:0*
T0*
_output_shapes

:@
my_sequential/dense_3/truediv_7RealDivmy_sequential/dense_3/Abs_2:y:0my_sequential/dense_3/Pow_2:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_3/add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense_3/add_4AddV2#my_sequential/dense_3/truediv_7:z:0&my_sequential/dense_3/add_4/y:output:0*
T0*
_output_shapes

:@p
my_sequential/dense_3/Floor_1Floormy_sequential/dense_3/add_4:z:0*
T0*
_output_shapes

:@c
my_sequential/dense_3/Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense_3/Less_1Less!my_sequential/dense_3/Floor_1:y:0'my_sequential/dense_3/Less_1/y:output:0*
T0*
_output_shapes

:@p
my_sequential/dense_3/Sign_1Sign!my_sequential/dense_3/truediv:z:0*
T0*
_output_shapes

:@
7my_sequential/dense_3/ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      l
'my_sequential/dense_3/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Æ
!my_sequential/dense_3/ones_like_1Fill@my_sequential/dense_3/ones_like_1/Shape/shape_as_tensor:output:00my_sequential/dense_3/ones_like_1/Const:output:0*
T0*
_output_shapes

:@b
my_sequential/dense_3/mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense_3/mul_5Mul*my_sequential/dense_3/ones_like_1:output:0&my_sequential/dense_3/mul_5/y:output:0*
T0*
_output_shapes

:@f
!my_sequential/dense_3/truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ 
my_sequential/dense_3/truediv_8RealDivmy_sequential/dense_3/mul_5:z:0*my_sequential/dense_3/truediv_8/y:output:0*
T0*
_output_shapes

:@¿
 my_sequential/dense_3/SelectV2_1SelectV2 my_sequential/dense_3/Less_1:z:0!my_sequential/dense_3/Floor_1:y:0#my_sequential/dense_3/truediv_8:z:0*
T0*
_output_shapes

:@
my_sequential/dense_3/mul_6Mul my_sequential/dense_3/Sign_1:y:0)my_sequential/dense_3/SelectV2_1:output:0*
T0*
_output_shapes

:@
my_sequential/dense_3/Mul_7Mul!my_sequential/dense_3/truediv:z:0my_sequential/dense_3/mul_6:z:0*
T0*
_output_shapes

:@x
.my_sequential/dense_3/Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¸
my_sequential/dense_3/Mean_2Meanmy_sequential/dense_3/Mul_7:z:07my_sequential/dense_3/Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
my_sequential/dense_3/Mul_8Mulmy_sequential/dense_3/mul_6:z:0my_sequential/dense_3/mul_6:z:0*
T0*
_output_shapes

:@x
.my_sequential/dense_3/Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¸
my_sequential/dense_3/Mean_3Meanmy_sequential/dense_3/Mul_8:z:07my_sequential/dense_3/Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(b
my_sequential/dense_3/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_3/add_5AddV2%my_sequential/dense_3/Mean_3:output:0&my_sequential/dense_3/add_5/y:output:0*
T0*
_output_shapes

:
my_sequential/dense_3/truediv_9RealDiv%my_sequential/dense_3/Mean_2:output:0my_sequential/dense_3/add_5:z:0*
T0*
_output_shapes

:b
my_sequential/dense_3/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_3/add_6AddV2#my_sequential/dense_3/truediv_9:z:0&my_sequential/dense_3/add_6/y:output:0*
T0*
_output_shapes

:l
my_sequential/dense_3/Log_2Logmy_sequential/dense_3/add_6:z:0*
T0*
_output_shapes

:g
"my_sequential/dense_3/truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?¢
 my_sequential/dense_3/truediv_10RealDivmy_sequential/dense_3/Log_2:y:0+my_sequential/dense_3/truediv_10/y:output:0*
T0*
_output_shapes

:u
my_sequential/dense_3/Round_2Round$my_sequential/dense_3/truediv_10:z:0*
T0*
_output_shapes

:b
my_sequential/dense_3/Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_3/Pow_3Pow&my_sequential/dense_3/Pow_3/x:output:0!my_sequential/dense_3/Round_2:y:0*
T0*
_output_shapes

:n
my_sequential/dense_3/Abs_3Abs!my_sequential/dense_3/truediv:z:0*
T0*
_output_shapes

:@
 my_sequential/dense_3/truediv_11RealDivmy_sequential/dense_3/Abs_3:y:0my_sequential/dense_3/Pow_3:z:0*
T0*
_output_shapes

:@b
my_sequential/dense_3/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense_3/add_7AddV2$my_sequential/dense_3/truediv_11:z:0&my_sequential/dense_3/add_7/y:output:0*
T0*
_output_shapes

:@p
my_sequential/dense_3/Floor_2Floormy_sequential/dense_3/add_7:z:0*
T0*
_output_shapes

:@c
my_sequential/dense_3/Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense_3/Less_2Less!my_sequential/dense_3/Floor_2:y:0'my_sequential/dense_3/Less_2/y:output:0*
T0*
_output_shapes

:@p
my_sequential/dense_3/Sign_2Sign!my_sequential/dense_3/truediv:z:0*
T0*
_output_shapes

:@
7my_sequential/dense_3/ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      l
'my_sequential/dense_3/ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Æ
!my_sequential/dense_3/ones_like_2Fill@my_sequential/dense_3/ones_like_2/Shape/shape_as_tensor:output:00my_sequential/dense_3/ones_like_2/Const:output:0*
T0*
_output_shapes

:@b
my_sequential/dense_3/mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A
my_sequential/dense_3/mul_9Mul*my_sequential/dense_3/ones_like_2:output:0&my_sequential/dense_3/mul_9/y:output:0*
T0*
_output_shapes

:@g
"my_sequential/dense_3/truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @¢
 my_sequential/dense_3/truediv_12RealDivmy_sequential/dense_3/mul_9:z:0+my_sequential/dense_3/truediv_12/y:output:0*
T0*
_output_shapes

:@À
 my_sequential/dense_3/SelectV2_2SelectV2 my_sequential/dense_3/Less_2:z:0!my_sequential/dense_3/Floor_2:y:0$my_sequential/dense_3/truediv_12:z:0*
T0*
_output_shapes

:@
my_sequential/dense_3/mul_10Mul my_sequential/dense_3/Sign_2:y:0)my_sequential/dense_3/SelectV2_2:output:0*
T0*
_output_shapes

:@
my_sequential/dense_3/Mul_11Mul!my_sequential/dense_3/truediv:z:0 my_sequential/dense_3/mul_10:z:0*
T0*
_output_shapes

:@x
.my_sequential/dense_3/Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_3/Mean_4Mean my_sequential/dense_3/Mul_11:z:07my_sequential/dense_3/Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
my_sequential/dense_3/Mul_12Mul my_sequential/dense_3/mul_10:z:0 my_sequential/dense_3/mul_10:z:0*
T0*
_output_shapes

:@x
.my_sequential/dense_3/Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_3/Mean_5Mean my_sequential/dense_3/Mul_12:z:07my_sequential/dense_3/Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(b
my_sequential/dense_3/add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_3/add_8AddV2%my_sequential/dense_3/Mean_5:output:0&my_sequential/dense_3/add_8/y:output:0*
T0*
_output_shapes

:
 my_sequential/dense_3/truediv_13RealDiv%my_sequential/dense_3/Mean_4:output:0my_sequential/dense_3/add_8:z:0*
T0*
_output_shapes

:b
my_sequential/dense_3/add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_3/add_9AddV2$my_sequential/dense_3/truediv_13:z:0&my_sequential/dense_3/add_9/y:output:0*
T0*
_output_shapes

:l
my_sequential/dense_3/Log_3Logmy_sequential/dense_3/add_9:z:0*
T0*
_output_shapes

:g
"my_sequential/dense_3/truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?¢
 my_sequential/dense_3/truediv_14RealDivmy_sequential/dense_3/Log_3:y:0+my_sequential/dense_3/truediv_14/y:output:0*
T0*
_output_shapes

:u
my_sequential/dense_3/Round_3Round$my_sequential/dense_3/truediv_14:z:0*
T0*
_output_shapes

:b
my_sequential/dense_3/Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_3/Pow_4Pow&my_sequential/dense_3/Pow_4/x:output:0!my_sequential/dense_3/Round_3:y:0*
T0*
_output_shapes

:n
my_sequential/dense_3/Abs_4Abs!my_sequential/dense_3/truediv:z:0*
T0*
_output_shapes

:@
 my_sequential/dense_3/truediv_15RealDivmy_sequential/dense_3/Abs_4:y:0my_sequential/dense_3/Pow_4:z:0*
T0*
_output_shapes

:@c
my_sequential/dense_3/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense_3/add_10AddV2$my_sequential/dense_3/truediv_15:z:0'my_sequential/dense_3/add_10/y:output:0*
T0*
_output_shapes

:@q
my_sequential/dense_3/Floor_3Floor my_sequential/dense_3/add_10:z:0*
T0*
_output_shapes

:@c
my_sequential/dense_3/Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense_3/Less_3Less!my_sequential/dense_3/Floor_3:y:0'my_sequential/dense_3/Less_3/y:output:0*
T0*
_output_shapes

:@p
my_sequential/dense_3/Sign_3Sign!my_sequential/dense_3/truediv:z:0*
T0*
_output_shapes

:@
7my_sequential/dense_3/ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      l
'my_sequential/dense_3/ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Æ
!my_sequential/dense_3/ones_like_3Fill@my_sequential/dense_3/ones_like_3/Shape/shape_as_tensor:output:00my_sequential/dense_3/ones_like_3/Const:output:0*
T0*
_output_shapes

:@c
my_sequential/dense_3/mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A¡
my_sequential/dense_3/mul_13Mul*my_sequential/dense_3/ones_like_3:output:0'my_sequential/dense_3/mul_13/y:output:0*
T0*
_output_shapes

:@g
"my_sequential/dense_3/truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @£
 my_sequential/dense_3/truediv_16RealDiv my_sequential/dense_3/mul_13:z:0+my_sequential/dense_3/truediv_16/y:output:0*
T0*
_output_shapes

:@À
 my_sequential/dense_3/SelectV2_3SelectV2 my_sequential/dense_3/Less_3:z:0!my_sequential/dense_3/Floor_3:y:0$my_sequential/dense_3/truediv_16:z:0*
T0*
_output_shapes

:@
my_sequential/dense_3/mul_14Mul my_sequential/dense_3/Sign_3:y:0)my_sequential/dense_3/SelectV2_3:output:0*
T0*
_output_shapes

:@
my_sequential/dense_3/Mul_15Mul!my_sequential/dense_3/truediv:z:0 my_sequential/dense_3/mul_14:z:0*
T0*
_output_shapes

:@x
.my_sequential/dense_3/Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_3/Mean_6Mean my_sequential/dense_3/Mul_15:z:07my_sequential/dense_3/Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
my_sequential/dense_3/Mul_16Mul my_sequential/dense_3/mul_14:z:0 my_sequential/dense_3/mul_14:z:0*
T0*
_output_shapes

:@x
.my_sequential/dense_3/Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_3/Mean_7Mean my_sequential/dense_3/Mul_16:z:07my_sequential/dense_3/Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(c
my_sequential/dense_3/add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_3/add_11AddV2%my_sequential/dense_3/Mean_7:output:0'my_sequential/dense_3/add_11/y:output:0*
T0*
_output_shapes

:
 my_sequential/dense_3/truediv_17RealDiv%my_sequential/dense_3/Mean_6:output:0 my_sequential/dense_3/add_11:z:0*
T0*
_output_shapes

:c
my_sequential/dense_3/add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_3/add_12AddV2$my_sequential/dense_3/truediv_17:z:0'my_sequential/dense_3/add_12/y:output:0*
T0*
_output_shapes

:m
my_sequential/dense_3/Log_4Log my_sequential/dense_3/add_12:z:0*
T0*
_output_shapes

:g
"my_sequential/dense_3/truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?¢
 my_sequential/dense_3/truediv_18RealDivmy_sequential/dense_3/Log_4:y:0+my_sequential/dense_3/truediv_18/y:output:0*
T0*
_output_shapes

:u
my_sequential/dense_3/Round_4Round$my_sequential/dense_3/truediv_18:z:0*
T0*
_output_shapes

:b
my_sequential/dense_3/Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_3/Pow_5Pow&my_sequential/dense_3/Pow_5/x:output:0!my_sequential/dense_3/Round_4:y:0*
T0*
_output_shapes

:n
my_sequential/dense_3/Abs_5Abs!my_sequential/dense_3/truediv:z:0*
T0*
_output_shapes

:@
 my_sequential/dense_3/truediv_19RealDivmy_sequential/dense_3/Abs_5:y:0my_sequential/dense_3/Pow_5:z:0*
T0*
_output_shapes

:@c
my_sequential/dense_3/add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
my_sequential/dense_3/add_13AddV2$my_sequential/dense_3/truediv_19:z:0'my_sequential/dense_3/add_13/y:output:0*
T0*
_output_shapes

:@q
my_sequential/dense_3/Floor_4Floor my_sequential/dense_3/add_13:z:0*
T0*
_output_shapes

:@c
my_sequential/dense_3/Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@
my_sequential/dense_3/Less_4Less!my_sequential/dense_3/Floor_4:y:0'my_sequential/dense_3/Less_4/y:output:0*
T0*
_output_shapes

:@p
my_sequential/dense_3/Sign_4Sign!my_sequential/dense_3/truediv:z:0*
T0*
_output_shapes

:@
7my_sequential/dense_3/ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      l
'my_sequential/dense_3/ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Æ
!my_sequential/dense_3/ones_like_4Fill@my_sequential/dense_3/ones_like_4/Shape/shape_as_tensor:output:00my_sequential/dense_3/ones_like_4/Const:output:0*
T0*
_output_shapes

:@c
my_sequential/dense_3/mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A¡
my_sequential/dense_3/mul_17Mul*my_sequential/dense_3/ones_like_4:output:0'my_sequential/dense_3/mul_17/y:output:0*
T0*
_output_shapes

:@g
"my_sequential/dense_3/truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @£
 my_sequential/dense_3/truediv_20RealDiv my_sequential/dense_3/mul_17:z:0+my_sequential/dense_3/truediv_20/y:output:0*
T0*
_output_shapes

:@À
 my_sequential/dense_3/SelectV2_4SelectV2 my_sequential/dense_3/Less_4:z:0!my_sequential/dense_3/Floor_4:y:0$my_sequential/dense_3/truediv_20:z:0*
T0*
_output_shapes

:@
my_sequential/dense_3/mul_18Mul my_sequential/dense_3/Sign_4:y:0)my_sequential/dense_3/SelectV2_4:output:0*
T0*
_output_shapes

:@
my_sequential/dense_3/Mul_19Mul!my_sequential/dense_3/truediv:z:0 my_sequential/dense_3/mul_18:z:0*
T0*
_output_shapes

:@x
.my_sequential/dense_3/Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_3/Mean_8Mean my_sequential/dense_3/Mul_19:z:07my_sequential/dense_3/Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
my_sequential/dense_3/Mul_20Mul my_sequential/dense_3/mul_18:z:0 my_sequential/dense_3/mul_18:z:0*
T0*
_output_shapes

:@x
.my_sequential/dense_3/Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¹
my_sequential/dense_3/Mean_9Mean my_sequential/dense_3/Mul_20:z:07my_sequential/dense_3/Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(c
my_sequential/dense_3/add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_3/add_14AddV2%my_sequential/dense_3/Mean_9:output:0'my_sequential/dense_3/add_14/y:output:0*
T0*
_output_shapes

:
 my_sequential/dense_3/truediv_21RealDiv%my_sequential/dense_3/Mean_8:output:0 my_sequential/dense_3/add_14:z:0*
T0*
_output_shapes

:c
my_sequential/dense_3/add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
my_sequential/dense_3/add_15AddV2$my_sequential/dense_3/truediv_21:z:0'my_sequential/dense_3/add_15/y:output:0*
T0*
_output_shapes

:m
my_sequential/dense_3/Log_5Log my_sequential/dense_3/add_15:z:0*
T0*
_output_shapes

:g
"my_sequential/dense_3/truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?¢
 my_sequential/dense_3/truediv_22RealDivmy_sequential/dense_3/Log_5:y:0+my_sequential/dense_3/truediv_22/y:output:0*
T0*
_output_shapes

:u
my_sequential/dense_3/Round_5Round$my_sequential/dense_3/truediv_22:z:0*
T0*
_output_shapes

:b
my_sequential/dense_3/Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
my_sequential/dense_3/Pow_6Pow&my_sequential/dense_3/Pow_6/x:output:0!my_sequential/dense_3/Round_5:y:0*
T0*
_output_shapes

:c
my_sequential/dense_3/mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
my_sequential/dense_3/mul_21Mulmy_sequential/dense_3/Pow_6:z:0'my_sequential/dense_3/mul_21/y:output:0*
T0*
_output_shapes

:
my_sequential/dense_3/mul_22Mulmy_sequential/dense_3/Cast:y:0!my_sequential/dense_3/truediv:z:0*
T0*
_output_shapes

:@
my_sequential/dense_3/mul_23Mulmy_sequential/dense_3/Cast:y:0 my_sequential/dense_3/mul_18:z:0*
T0*
_output_shapes

:@g
"my_sequential/dense_3/truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A£
 my_sequential/dense_3/truediv_23RealDiv my_sequential/dense_3/mul_23:z:0+my_sequential/dense_3/truediv_23/y:output:0*
T0*
_output_shapes

:@
my_sequential/dense_3/mul_24Mul my_sequential/dense_3/mul_21:z:0$my_sequential/dense_3/truediv_23:z:0*
T0*
_output_shapes

:@k
my_sequential/dense_3/NegNeg my_sequential/dense_3/mul_22:z:0*
T0*
_output_shapes

:@
my_sequential/dense_3/add_16AddV2my_sequential/dense_3/Neg:y:0 my_sequential/dense_3/mul_24:z:0*
T0*
_output_shapes

:@c
my_sequential/dense_3/mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/dense_3/mul_25Mul'my_sequential/dense_3/mul_25/x:output:0 my_sequential/dense_3/add_16:z:0*
T0*
_output_shapes

:@}
"my_sequential/dense_3/StopGradientStopGradient my_sequential/dense_3/mul_25:z:0*
T0*
_output_shapes

:@
my_sequential/dense_3/add_17AddV2 my_sequential/dense_3/mul_22:z:0+my_sequential/dense_3/StopGradient:output:0*
T0*
_output_shapes

:@
my_sequential/dense_3/MatMulMatMulmy_sequential/re_lu_2/add_2:z:0 my_sequential/dense_3/add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
my_sequential/dense_3/Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :_
my_sequential/dense_3/Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : 
my_sequential/dense_3/Pow_7Pow&my_sequential/dense_3/Pow_7/x:output:0&my_sequential/dense_3/Pow_7/y:output:0*
T0*
_output_shapes
: u
my_sequential/dense_3/Cast_1Castmy_sequential/dense_3/Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: 
&my_sequential/dense_3/ReadVariableOp_1ReadVariableOp/my_sequential_dense_3_readvariableop_1_resource*
_output_shapes
:*
dtype0c
my_sequential/dense_3/mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A¡
my_sequential/dense_3/mul_26Mul.my_sequential/dense_3/ReadVariableOp_1:value:0'my_sequential/dense_3/mul_26/y:output:0*
T0*
_output_shapes
:
 my_sequential/dense_3/truediv_24RealDiv my_sequential/dense_3/mul_26:z:0 my_sequential/dense_3/Cast_1:y:0*
T0*
_output_shapes
:m
my_sequential/dense_3/Neg_1Neg$my_sequential/dense_3/truediv_24:z:0*
T0*
_output_shapes
:q
my_sequential/dense_3/Round_6Round$my_sequential/dense_3/truediv_24:z:0*
T0*
_output_shapes
:
my_sequential/dense_3/add_18AddV2my_sequential/dense_3/Neg_1:y:0!my_sequential/dense_3/Round_6:y:0*
T0*
_output_shapes
:{
$my_sequential/dense_3/StopGradient_1StopGradient my_sequential/dense_3/add_18:z:0*
T0*
_output_shapes
:
my_sequential/dense_3/add_19AddV2$my_sequential/dense_3/truediv_24:z:0-my_sequential/dense_3/StopGradient_1:output:0*
T0*
_output_shapes
:r
-my_sequential/dense_3/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@µ
+my_sequential/dense_3/clip_by_value/MinimumMinimum my_sequential/dense_3/add_19:z:06my_sequential/dense_3/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:j
%my_sequential/dense_3/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Á´
#my_sequential/dense_3/clip_by_valueMaximum/my_sequential/dense_3/clip_by_value/Minimum:z:0.my_sequential/dense_3/clip_by_value/y:output:0*
T0*
_output_shapes
:
my_sequential/dense_3/mul_27Mul my_sequential/dense_3/Cast_1:y:0'my_sequential/dense_3/clip_by_value:z:0*
T0*
_output_shapes
:g
"my_sequential/dense_3/truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A
 my_sequential/dense_3/truediv_25RealDiv my_sequential/dense_3/mul_27:z:0+my_sequential/dense_3/truediv_25/y:output:0*
T0*
_output_shapes
:c
my_sequential/dense_3/mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/dense_3/mul_28Mul'my_sequential/dense_3/mul_28/x:output:0$my_sequential/dense_3/truediv_25:z:0*
T0*
_output_shapes
:
&my_sequential/dense_3/ReadVariableOp_2ReadVariableOp/my_sequential_dense_3_readvariableop_1_resource*
_output_shapes
:*
dtype0w
my_sequential/dense_3/Neg_2Neg.my_sequential/dense_3/ReadVariableOp_2:value:0*
T0*
_output_shapes
:
my_sequential/dense_3/add_20AddV2my_sequential/dense_3/Neg_2:y:0 my_sequential/dense_3/mul_28:z:0*
T0*
_output_shapes
:c
my_sequential/dense_3/mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
my_sequential/dense_3/mul_29Mul'my_sequential/dense_3/mul_29/x:output:0 my_sequential/dense_3/add_20:z:0*
T0*
_output_shapes
:{
$my_sequential/dense_3/StopGradient_2StopGradient my_sequential/dense_3/mul_29:z:0*
T0*
_output_shapes
:
&my_sequential/dense_3/ReadVariableOp_3ReadVariableOp/my_sequential_dense_3_readvariableop_1_resource*
_output_shapes
:*
dtype0©
my_sequential/dense_3/add_21AddV2.my_sequential/dense_3/ReadVariableOp_3:value:0-my_sequential/dense_3/StopGradient_2:output:0*
T0*
_output_shapes
:¤
my_sequential/dense_3/BiasAddBiasAdd&my_sequential/dense_3/MatMul:product:0 my_sequential/dense_3/add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
IdentityIdentity&my_sequential/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
NoOpNoOp;^my_sequential/batch_normalization/batchnorm/ReadVariableOp=^my_sequential/batch_normalization/batchnorm/ReadVariableOp_1=^my_sequential/batch_normalization/batchnorm/ReadVariableOp_2?^my_sequential/batch_normalization/batchnorm/mul/ReadVariableOp=^my_sequential/batch_normalization_1/batchnorm/ReadVariableOp?^my_sequential/batch_normalization_1/batchnorm/ReadVariableOp_1?^my_sequential/batch_normalization_1/batchnorm/ReadVariableOp_2A^my_sequential/batch_normalization_1/batchnorm/mul/ReadVariableOp=^my_sequential/batch_normalization_2/batchnorm/ReadVariableOp?^my_sequential/batch_normalization_2/batchnorm/ReadVariableOp_1?^my_sequential/batch_normalization_2/batchnorm/ReadVariableOp_2A^my_sequential/batch_normalization_2/batchnorm/mul/ReadVariableOp#^my_sequential/dense/ReadVariableOp%^my_sequential/dense/ReadVariableOp_1%^my_sequential/dense/ReadVariableOp_2%^my_sequential/dense/ReadVariableOp_3%^my_sequential/dense_1/ReadVariableOp'^my_sequential/dense_1/ReadVariableOp_1'^my_sequential/dense_1/ReadVariableOp_2'^my_sequential/dense_1/ReadVariableOp_3%^my_sequential/dense_2/ReadVariableOp'^my_sequential/dense_2/ReadVariableOp_1'^my_sequential/dense_2/ReadVariableOp_2'^my_sequential/dense_2/ReadVariableOp_3%^my_sequential/dense_3/ReadVariableOp'^my_sequential/dense_3/ReadVariableOp_1'^my_sequential/dense_3/ReadVariableOp_2'^my_sequential/dense_3/ReadVariableOp_3*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿv: : : : : : : : : : : : : : : : : : : : 2x
:my_sequential/batch_normalization/batchnorm/ReadVariableOp:my_sequential/batch_normalization/batchnorm/ReadVariableOp2|
<my_sequential/batch_normalization/batchnorm/ReadVariableOp_1<my_sequential/batch_normalization/batchnorm/ReadVariableOp_12|
<my_sequential/batch_normalization/batchnorm/ReadVariableOp_2<my_sequential/batch_normalization/batchnorm/ReadVariableOp_22
>my_sequential/batch_normalization/batchnorm/mul/ReadVariableOp>my_sequential/batch_normalization/batchnorm/mul/ReadVariableOp2|
<my_sequential/batch_normalization_1/batchnorm/ReadVariableOp<my_sequential/batch_normalization_1/batchnorm/ReadVariableOp2
>my_sequential/batch_normalization_1/batchnorm/ReadVariableOp_1>my_sequential/batch_normalization_1/batchnorm/ReadVariableOp_12
>my_sequential/batch_normalization_1/batchnorm/ReadVariableOp_2>my_sequential/batch_normalization_1/batchnorm/ReadVariableOp_22
@my_sequential/batch_normalization_1/batchnorm/mul/ReadVariableOp@my_sequential/batch_normalization_1/batchnorm/mul/ReadVariableOp2|
<my_sequential/batch_normalization_2/batchnorm/ReadVariableOp<my_sequential/batch_normalization_2/batchnorm/ReadVariableOp2
>my_sequential/batch_normalization_2/batchnorm/ReadVariableOp_1>my_sequential/batch_normalization_2/batchnorm/ReadVariableOp_12
>my_sequential/batch_normalization_2/batchnorm/ReadVariableOp_2>my_sequential/batch_normalization_2/batchnorm/ReadVariableOp_22
@my_sequential/batch_normalization_2/batchnorm/mul/ReadVariableOp@my_sequential/batch_normalization_2/batchnorm/mul/ReadVariableOp2H
"my_sequential/dense/ReadVariableOp"my_sequential/dense/ReadVariableOp2L
$my_sequential/dense/ReadVariableOp_1$my_sequential/dense/ReadVariableOp_12L
$my_sequential/dense/ReadVariableOp_2$my_sequential/dense/ReadVariableOp_22L
$my_sequential/dense/ReadVariableOp_3$my_sequential/dense/ReadVariableOp_32L
$my_sequential/dense_1/ReadVariableOp$my_sequential/dense_1/ReadVariableOp2P
&my_sequential/dense_1/ReadVariableOp_1&my_sequential/dense_1/ReadVariableOp_12P
&my_sequential/dense_1/ReadVariableOp_2&my_sequential/dense_1/ReadVariableOp_22P
&my_sequential/dense_1/ReadVariableOp_3&my_sequential/dense_1/ReadVariableOp_32L
$my_sequential/dense_2/ReadVariableOp$my_sequential/dense_2/ReadVariableOp2P
&my_sequential/dense_2/ReadVariableOp_1&my_sequential/dense_2/ReadVariableOp_12P
&my_sequential/dense_2/ReadVariableOp_2&my_sequential/dense_2/ReadVariableOp_22P
&my_sequential/dense_2/ReadVariableOp_3&my_sequential/dense_2/ReadVariableOp_32L
$my_sequential/dense_3/ReadVariableOp$my_sequential/dense_3/ReadVariableOp2P
&my_sequential/dense_3/ReadVariableOp_1&my_sequential/dense_3/ReadVariableOp_12P
&my_sequential/dense_3/ReadVariableOp_2&my_sequential/dense_3/ReadVariableOp_22P
&my_sequential/dense_3/ReadVariableOp_3&my_sequential/dense_3/ReadVariableOp_3:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!
_user_specified_name	input_1
¾

\
@__inference_re_lu_layer_call_and_return_conditional_losses_75687

inputs
identityF
SignSigninputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
AbsAbsSign:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
subSubsub/x:output:0Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
addAddV2Sign:y:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
TanhTanhinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
NegNegTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mulMulmul/x:output:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
add_1AddV2Neg:y:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
StopGradientStopGradient	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
add_2AddV2Tanh:y:0StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ï
b
)__inference_dropout_2_layer_call_fn_80369

inputs
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_76581o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ë
­
N__inference_batch_normalization_layer_call_and_return_conditional_losses_75178

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ò	
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_76581

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Í
¯
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_79946

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

E
)__inference_dropout_2_layer_call_fn_80364

inputs
identity¯
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_76233`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
À

^
B__inference_re_lu_2_layer_call_and_return_conditional_losses_76251

inputs
identityF
SignSigninputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
AbsAbsSign:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
subSubsub/x:output:0Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
addAddV2Sign:y:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
TanhTanhinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
NegNegTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mulMulmul/x:output:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
add_1AddV2Neg:y:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
StopGradientStopGradient	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
add_2AddV2Tanh:y:0StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

C
'__inference_re_lu_2_layer_call_fn_80391

inputs
identity­
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_76251`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
r
¦
__inference__traced_save_80852
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Î
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*÷
valueíBê:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHâ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B À
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*û
_input_shapesé
æ: :v@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@:: : : : : : : : : :v@:@:@:@:@@:@:@:@:@@:@:@:@:@::v@:@:@:@:@@:@:@:@:@@:@:@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:v@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:v@: 

_output_shapes
:@:  

_output_shapes
:@: !

_output_shapes
:@:$" 

_output_shapes

:@@: #

_output_shapes
:@: $

_output_shapes
:@: %

_output_shapes
:@:$& 

_output_shapes

:@@: '

_output_shapes
:@: (

_output_shapes
:@: )

_output_shapes
:@:$* 

_output_shapes

:@: +

_output_shapes
::$, 

_output_shapes

:v@: -

_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@:$0 

_output_shapes

:@@: 1

_output_shapes
:@: 2

_output_shapes
:@: 3

_output_shapes
:@:$4 

_output_shapes

:@@: 5

_output_shapes
:@: 6

_output_shapes
:@: 7

_output_shapes
:@:$8 

_output_shapes

:@: 9

_output_shapes
:::

_output_shapes
: 
è¨
ÿ
B__inference_dense_2_layer_call_and_return_conditional_losses_76213

inputs)
readvariableop_resource:@@'
readvariableop_1_resource:@
identity¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@@*
dtype0]
truedivRealDivReadVariableOp:value:0Cast:y:0*
T0*
_output_shapes

:@@@
AbsAbstruediv:z:0*
T0*
_output_shapes

:@@_
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: m
MaxMaxAbs:y:0Max/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Q
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes

:@P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A\
	truediv_1RealDivmul:z:0truediv_1/y:output:0*
T0*
_output_shapes

:@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3T
addAddV2truediv_1:z:0add/y:output:0*
T0*
_output_shapes

:@<
LogLogadd:z:0*
T0*
_output_shapes

:@P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?\
	truediv_2RealDivLog:y:0truediv_2/y:output:0*
T0*
_output_shapes

:@F
RoundRoundtruediv_2:z:0*
T0*
_output_shapes

:@L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @R
Pow_1PowPow_1/x:output:0	Round:y:0*
T0*
_output_shapes

:@B
Abs_1Abstruediv:z:0*
T0*
_output_shapes

:@@S
	truediv_3RealDiv	Abs_1:y:0	Pow_1:z:0*
T0*
_output_shapes

:@@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
add_1AddV2truediv_3:z:0add_1/y:output:0*
T0*
_output_shapes

:@@B
FloorFloor	add_1:z:0*
T0*
_output_shapes

:@@K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@Q
LessLess	Floor:y:0Less/y:output:0*
T0*
_output_shapes

:@@B
SignSigntruediv:z:0*
T0*
_output_shapes

:@@p
ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
	ones_likeFill(ones_like/Shape/shape_as_tensor:output:0ones_like/Const:output:0*
T0*
_output_shapes

:@@L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A[
mul_1Mulones_like:output:0mul_1/y:output:0*
T0*
_output_shapes

:@@P
truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
	truediv_4RealDiv	mul_1:z:0truediv_4/y:output:0*
T0*
_output_shapes

:@@a
SelectV2SelectV2Less:z:0	Floor:y:0truediv_4:z:0*
T0*
_output_shapes

:@@R
mul_2MulSign:y:0SelectV2:output:0*
T0*
_output_shapes

:@@M
Mul_3Multruediv:z:0	mul_2:z:0*
T0*
_output_shapes

:@@`
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: r
MeanMean	Mul_3:z:0Mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(K
Mul_4Mul	mul_2:z:0	mul_2:z:0*
T0*
_output_shapes

:@@b
Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_1Mean	Mul_4:z:0!Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_2AddV2Mean_1:output:0add_2/y:output:0*
T0*
_output_shapes

:@W
	truediv_5RealDivMean:output:0	add_2:z:0*
T0*
_output_shapes

:@L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3X
add_3AddV2truediv_5:z:0add_3/y:output:0*
T0*
_output_shapes

:@@
Log_1Log	add_3:z:0*
T0*
_output_shapes

:@P
truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?^
	truediv_6RealDiv	Log_1:y:0truediv_6/y:output:0*
T0*
_output_shapes

:@H
Round_1Roundtruediv_6:z:0*
T0*
_output_shapes

:@L
Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_2PowPow_2/x:output:0Round_1:y:0*
T0*
_output_shapes

:@B
Abs_2Abstruediv:z:0*
T0*
_output_shapes

:@@S
	truediv_7RealDiv	Abs_2:y:0	Pow_2:z:0*
T0*
_output_shapes

:@@L
add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
add_4AddV2truediv_7:z:0add_4/y:output:0*
T0*
_output_shapes

:@@D
Floor_1Floor	add_4:z:0*
T0*
_output_shapes

:@@M
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_1LessFloor_1:y:0Less_1/y:output:0*
T0*
_output_shapes

:@@D
Sign_1Signtruediv:z:0*
T0*
_output_shapes

:@@r
!ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_1Fill*ones_like_1/Shape/shape_as_tensor:output:0ones_like_1/Const:output:0*
T0*
_output_shapes

:@@L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A]
mul_5Mulones_like_1:output:0mul_5/y:output:0*
T0*
_output_shapes

:@@P
truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
	truediv_8RealDiv	mul_5:z:0truediv_8/y:output:0*
T0*
_output_shapes

:@@g

SelectV2_1SelectV2
Less_1:z:0Floor_1:y:0truediv_8:z:0*
T0*
_output_shapes

:@@V
mul_6Mul
Sign_1:y:0SelectV2_1:output:0*
T0*
_output_shapes

:@@M
Mul_7Multruediv:z:0	mul_6:z:0*
T0*
_output_shapes

:@@b
Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_2Mean	Mul_7:z:0!Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(K
Mul_8Mul	mul_6:z:0	mul_6:z:0*
T0*
_output_shapes

:@@b
Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_3Mean	Mul_8:z:0!Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_5AddV2Mean_3:output:0add_5/y:output:0*
T0*
_output_shapes

:@Y
	truediv_9RealDivMean_2:output:0	add_5:z:0*
T0*
_output_shapes

:@L
add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3X
add_6AddV2truediv_9:z:0add_6/y:output:0*
T0*
_output_shapes

:@@
Log_2Log	add_6:z:0*
T0*
_output_shapes

:@Q
truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_10RealDiv	Log_2:y:0truediv_10/y:output:0*
T0*
_output_shapes

:@I
Round_2Roundtruediv_10:z:0*
T0*
_output_shapes

:@L
Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_3PowPow_3/x:output:0Round_2:y:0*
T0*
_output_shapes

:@B
Abs_3Abstruediv:z:0*
T0*
_output_shapes

:@@T

truediv_11RealDiv	Abs_3:y:0	Pow_3:z:0*
T0*
_output_shapes

:@@L
add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Y
add_7AddV2truediv_11:z:0add_7/y:output:0*
T0*
_output_shapes

:@@D
Floor_2Floor	add_7:z:0*
T0*
_output_shapes

:@@M
Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_2LessFloor_2:y:0Less_2/y:output:0*
T0*
_output_shapes

:@@D
Sign_2Signtruediv:z:0*
T0*
_output_shapes

:@@r
!ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_2Fill*ones_like_2/Shape/shape_as_tensor:output:0ones_like_2/Const:output:0*
T0*
_output_shapes

:@@L
mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A]
mul_9Mulones_like_2:output:0mul_9/y:output:0*
T0*
_output_shapes

:@@Q
truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @`

truediv_12RealDiv	mul_9:z:0truediv_12/y:output:0*
T0*
_output_shapes

:@@h

SelectV2_2SelectV2
Less_2:z:0Floor_2:y:0truediv_12:z:0*
T0*
_output_shapes

:@@W
mul_10Mul
Sign_2:y:0SelectV2_2:output:0*
T0*
_output_shapes

:@@O
Mul_11Multruediv:z:0
mul_10:z:0*
T0*
_output_shapes

:@@b
Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_4Mean
Mul_11:z:0!Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_12Mul
mul_10:z:0
mul_10:z:0*
T0*
_output_shapes

:@@b
Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_5Mean
Mul_12:z:0!Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_8AddV2Mean_5:output:0add_8/y:output:0*
T0*
_output_shapes

:@Z

truediv_13RealDivMean_4:output:0	add_8:z:0*
T0*
_output_shapes

:@L
add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Y
add_9AddV2truediv_13:z:0add_9/y:output:0*
T0*
_output_shapes

:@@
Log_3Log	add_9:z:0*
T0*
_output_shapes

:@Q
truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_14RealDiv	Log_3:y:0truediv_14/y:output:0*
T0*
_output_shapes

:@I
Round_3Roundtruediv_14:z:0*
T0*
_output_shapes

:@L
Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_4PowPow_4/x:output:0Round_3:y:0*
T0*
_output_shapes

:@B
Abs_4Abstruediv:z:0*
T0*
_output_shapes

:@@T

truediv_15RealDiv	Abs_4:y:0	Pow_4:z:0*
T0*
_output_shapes

:@@M
add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_10AddV2truediv_15:z:0add_10/y:output:0*
T0*
_output_shapes

:@@E
Floor_3Floor
add_10:z:0*
T0*
_output_shapes

:@@M
Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_3LessFloor_3:y:0Less_3/y:output:0*
T0*
_output_shapes

:@@D
Sign_3Signtruediv:z:0*
T0*
_output_shapes

:@@r
!ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   V
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_3Fill*ones_like_3/Shape/shape_as_tensor:output:0ones_like_3/Const:output:0*
T0*
_output_shapes

:@@M
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A_
mul_13Mulones_like_3:output:0mul_13/y:output:0*
T0*
_output_shapes

:@@Q
truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @a

truediv_16RealDiv
mul_13:z:0truediv_16/y:output:0*
T0*
_output_shapes

:@@h

SelectV2_3SelectV2
Less_3:z:0Floor_3:y:0truediv_16:z:0*
T0*
_output_shapes

:@@W
mul_14Mul
Sign_3:y:0SelectV2_3:output:0*
T0*
_output_shapes

:@@O
Mul_15Multruediv:z:0
mul_14:z:0*
T0*
_output_shapes

:@@b
Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_6Mean
Mul_15:z:0!Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_16Mul
mul_14:z:0
mul_14:z:0*
T0*
_output_shapes

:@@b
Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_7Mean
Mul_16:z:0!Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(M
add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3\
add_11AddV2Mean_7:output:0add_11/y:output:0*
T0*
_output_shapes

:@[

truediv_17RealDivMean_6:output:0
add_11:z:0*
T0*
_output_shapes

:@M
add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3[
add_12AddV2truediv_17:z:0add_12/y:output:0*
T0*
_output_shapes

:@A
Log_4Log
add_12:z:0*
T0*
_output_shapes

:@Q
truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_18RealDiv	Log_4:y:0truediv_18/y:output:0*
T0*
_output_shapes

:@I
Round_4Roundtruediv_18:z:0*
T0*
_output_shapes

:@L
Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_5PowPow_5/x:output:0Round_4:y:0*
T0*
_output_shapes

:@B
Abs_5Abstruediv:z:0*
T0*
_output_shapes

:@@T

truediv_19RealDiv	Abs_5:y:0	Pow_5:z:0*
T0*
_output_shapes

:@@M
add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_13AddV2truediv_19:z:0add_13/y:output:0*
T0*
_output_shapes

:@@E
Floor_4Floor
add_13:z:0*
T0*
_output_shapes

:@@M
Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_4LessFloor_4:y:0Less_4/y:output:0*
T0*
_output_shapes

:@@D
Sign_4Signtruediv:z:0*
T0*
_output_shapes

:@@r
!ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   V
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_4Fill*ones_like_4/Shape/shape_as_tensor:output:0ones_like_4/Const:output:0*
T0*
_output_shapes

:@@M
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A_
mul_17Mulones_like_4:output:0mul_17/y:output:0*
T0*
_output_shapes

:@@Q
truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @a

truediv_20RealDiv
mul_17:z:0truediv_20/y:output:0*
T0*
_output_shapes

:@@h

SelectV2_4SelectV2
Less_4:z:0Floor_4:y:0truediv_20:z:0*
T0*
_output_shapes

:@@W
mul_18Mul
Sign_4:y:0SelectV2_4:output:0*
T0*
_output_shapes

:@@O
Mul_19Multruediv:z:0
mul_18:z:0*
T0*
_output_shapes

:@@b
Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_8Mean
Mul_19:z:0!Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_20Mul
mul_18:z:0
mul_18:z:0*
T0*
_output_shapes

:@@b
Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_9Mean
Mul_20:z:0!Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(M
add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3\
add_14AddV2Mean_9:output:0add_14/y:output:0*
T0*
_output_shapes

:@[

truediv_21RealDivMean_8:output:0
add_14:z:0*
T0*
_output_shapes

:@M
add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3[
add_15AddV2truediv_21:z:0add_15/y:output:0*
T0*
_output_shapes

:@A
Log_5Log
add_15:z:0*
T0*
_output_shapes

:@Q
truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_22RealDiv	Log_5:y:0truediv_22/y:output:0*
T0*
_output_shapes

:@I
Round_5Roundtruediv_22:z:0*
T0*
_output_shapes

:@L
Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_6PowPow_6/x:output:0Round_5:y:0*
T0*
_output_shapes

:@M
mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AT
mul_21Mul	Pow_6:z:0mul_21/y:output:0*
T0*
_output_shapes

:@M
mul_22MulCast:y:0truediv:z:0*
T0*
_output_shapes

:@@L
mul_23MulCast:y:0
mul_18:z:0*
T0*
_output_shapes

:@@Q
truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aa

truediv_23RealDiv
mul_23:z:0truediv_23/y:output:0*
T0*
_output_shapes

:@@R
mul_24Mul
mul_21:z:0truediv_23:z:0*
T0*
_output_shapes

:@@?
NegNeg
mul_22:z:0*
T0*
_output_shapes

:@@M
add_16AddV2Neg:y:0
mul_24:z:0*
T0*
_output_shapes

:@@M
mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mul_25Mulmul_25/x:output:0
add_16:z:0*
T0*
_output_shapes

:@@Q
StopGradientStopGradient
mul_25:z:0*
T0*
_output_shapes

:@@[
add_17AddV2
mul_22:z:0StopGradient:output:0*
T0*
_output_shapes

:@@V
MatMulMatMulinputs
add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@I
Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_7PowPow_7/x:output:0Pow_7/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0M
mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A_
mul_26MulReadVariableOp_1:value:0mul_26/y:output:0*
T0*
_output_shapes
:@R

truediv_24RealDiv
mul_26:z:0
Cast_1:y:0*
T0*
_output_shapes
:@A
Neg_1Negtruediv_24:z:0*
T0*
_output_shapes
:@E
Round_6Roundtruediv_24:z:0*
T0*
_output_shapes
:@L
add_18AddV2	Neg_1:y:0Round_6:y:0*
T0*
_output_shapes
:@O
StopGradient_1StopGradient
add_18:z:0*
T0*
_output_shapes
:@]
add_19AddV2truediv_24:z:0StopGradient_1:output:0*
T0*
_output_shapes
:@\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@s
clip_by_value/MinimumMinimum
add_19:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:@T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ár
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:@Q
mul_27Mul
Cast_1:y:0clip_by_value:z:0*
T0*
_output_shapes
:@Q
truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A]

truediv_25RealDiv
mul_27:z:0truediv_25/y:output:0*
T0*
_output_shapes
:@M
mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mul_28Mulmul_28/x:output:0truediv_25:z:0*
T0*
_output_shapes
:@f
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0K
Neg_2NegReadVariableOp_2:value:0*
T0*
_output_shapes
:@K
add_20AddV2	Neg_2:y:0
mul_28:z:0*
T0*
_output_shapes
:@M
mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Q
mul_29Mulmul_29/x:output:0
add_20:z:0*
T0*
_output_shapes
:@O
StopGradient_2StopGradient
mul_29:z:0*
T0*
_output_shapes
:@f
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0g
add_21AddV2ReadVariableOp_3:value:0StopGradient_2:output:0*
T0*
_output_shapes
:@b
BiasAddBiasAddMatMul:product:0
add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_3:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 
Î
3__inference_batch_normalization_layer_call_fn_79547

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_75225o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
×
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_76233

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
è¨
ÿ
B__inference_dense_3_layer_call_and_return_conditional_losses_80658

inputs)
readvariableop_resource:@'
readvariableop_1_resource:
identity¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@*
dtype0]
truedivRealDivReadVariableOp:value:0Cast:y:0*
T0*
_output_shapes

:@@
AbsAbstruediv:z:0*
T0*
_output_shapes

:@_
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: m
MaxMaxAbs:y:0Max/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Q
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A\
	truediv_1RealDivmul:z:0truediv_1/y:output:0*
T0*
_output_shapes

:J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3T
addAddV2truediv_1:z:0add/y:output:0*
T0*
_output_shapes

:<
LogLogadd:z:0*
T0*
_output_shapes

:P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?\
	truediv_2RealDivLog:y:0truediv_2/y:output:0*
T0*
_output_shapes

:F
RoundRoundtruediv_2:z:0*
T0*
_output_shapes

:L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @R
Pow_1PowPow_1/x:output:0	Round:y:0*
T0*
_output_shapes

:B
Abs_1Abstruediv:z:0*
T0*
_output_shapes

:@S
	truediv_3RealDiv	Abs_1:y:0	Pow_1:z:0*
T0*
_output_shapes

:@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
add_1AddV2truediv_3:z:0add_1/y:output:0*
T0*
_output_shapes

:@B
FloorFloor	add_1:z:0*
T0*
_output_shapes

:@K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@Q
LessLess	Floor:y:0Less/y:output:0*
T0*
_output_shapes

:@B
SignSigntruediv:z:0*
T0*
_output_shapes

:@p
ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
	ones_likeFill(ones_like/Shape/shape_as_tensor:output:0ones_like/Const:output:0*
T0*
_output_shapes

:@L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A[
mul_1Mulones_like:output:0mul_1/y:output:0*
T0*
_output_shapes

:@P
truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
	truediv_4RealDiv	mul_1:z:0truediv_4/y:output:0*
T0*
_output_shapes

:@a
SelectV2SelectV2Less:z:0	Floor:y:0truediv_4:z:0*
T0*
_output_shapes

:@R
mul_2MulSign:y:0SelectV2:output:0*
T0*
_output_shapes

:@M
Mul_3Multruediv:z:0	mul_2:z:0*
T0*
_output_shapes

:@`
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: r
MeanMean	Mul_3:z:0Mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(K
Mul_4Mul	mul_2:z:0	mul_2:z:0*
T0*
_output_shapes

:@b
Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_1Mean	Mul_4:z:0!Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_2AddV2Mean_1:output:0add_2/y:output:0*
T0*
_output_shapes

:W
	truediv_5RealDivMean:output:0	add_2:z:0*
T0*
_output_shapes

:L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3X
add_3AddV2truediv_5:z:0add_3/y:output:0*
T0*
_output_shapes

:@
Log_1Log	add_3:z:0*
T0*
_output_shapes

:P
truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?^
	truediv_6RealDiv	Log_1:y:0truediv_6/y:output:0*
T0*
_output_shapes

:H
Round_1Roundtruediv_6:z:0*
T0*
_output_shapes

:L
Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_2PowPow_2/x:output:0Round_1:y:0*
T0*
_output_shapes

:B
Abs_2Abstruediv:z:0*
T0*
_output_shapes

:@S
	truediv_7RealDiv	Abs_2:y:0	Pow_2:z:0*
T0*
_output_shapes

:@L
add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
add_4AddV2truediv_7:z:0add_4/y:output:0*
T0*
_output_shapes

:@D
Floor_1Floor	add_4:z:0*
T0*
_output_shapes

:@M
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_1LessFloor_1:y:0Less_1/y:output:0*
T0*
_output_shapes

:@D
Sign_1Signtruediv:z:0*
T0*
_output_shapes

:@r
!ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_1Fill*ones_like_1/Shape/shape_as_tensor:output:0ones_like_1/Const:output:0*
T0*
_output_shapes

:@L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A]
mul_5Mulones_like_1:output:0mul_5/y:output:0*
T0*
_output_shapes

:@P
truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
	truediv_8RealDiv	mul_5:z:0truediv_8/y:output:0*
T0*
_output_shapes

:@g

SelectV2_1SelectV2
Less_1:z:0Floor_1:y:0truediv_8:z:0*
T0*
_output_shapes

:@V
mul_6Mul
Sign_1:y:0SelectV2_1:output:0*
T0*
_output_shapes

:@M
Mul_7Multruediv:z:0	mul_6:z:0*
T0*
_output_shapes

:@b
Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_2Mean	Mul_7:z:0!Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(K
Mul_8Mul	mul_6:z:0	mul_6:z:0*
T0*
_output_shapes

:@b
Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_3Mean	Mul_8:z:0!Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_5AddV2Mean_3:output:0add_5/y:output:0*
T0*
_output_shapes

:Y
	truediv_9RealDivMean_2:output:0	add_5:z:0*
T0*
_output_shapes

:L
add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3X
add_6AddV2truediv_9:z:0add_6/y:output:0*
T0*
_output_shapes

:@
Log_2Log	add_6:z:0*
T0*
_output_shapes

:Q
truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_10RealDiv	Log_2:y:0truediv_10/y:output:0*
T0*
_output_shapes

:I
Round_2Roundtruediv_10:z:0*
T0*
_output_shapes

:L
Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_3PowPow_3/x:output:0Round_2:y:0*
T0*
_output_shapes

:B
Abs_3Abstruediv:z:0*
T0*
_output_shapes

:@T

truediv_11RealDiv	Abs_3:y:0	Pow_3:z:0*
T0*
_output_shapes

:@L
add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Y
add_7AddV2truediv_11:z:0add_7/y:output:0*
T0*
_output_shapes

:@D
Floor_2Floor	add_7:z:0*
T0*
_output_shapes

:@M
Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_2LessFloor_2:y:0Less_2/y:output:0*
T0*
_output_shapes

:@D
Sign_2Signtruediv:z:0*
T0*
_output_shapes

:@r
!ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_2Fill*ones_like_2/Shape/shape_as_tensor:output:0ones_like_2/Const:output:0*
T0*
_output_shapes

:@L
mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A]
mul_9Mulones_like_2:output:0mul_9/y:output:0*
T0*
_output_shapes

:@Q
truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @`

truediv_12RealDiv	mul_9:z:0truediv_12/y:output:0*
T0*
_output_shapes

:@h

SelectV2_2SelectV2
Less_2:z:0Floor_2:y:0truediv_12:z:0*
T0*
_output_shapes

:@W
mul_10Mul
Sign_2:y:0SelectV2_2:output:0*
T0*
_output_shapes

:@O
Mul_11Multruediv:z:0
mul_10:z:0*
T0*
_output_shapes

:@b
Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_4Mean
Mul_11:z:0!Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(N
Mul_12Mul
mul_10:z:0
mul_10:z:0*
T0*
_output_shapes

:@b
Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_5Mean
Mul_12:z:0!Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(L
add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_8AddV2Mean_5:output:0add_8/y:output:0*
T0*
_output_shapes

:Z

truediv_13RealDivMean_4:output:0	add_8:z:0*
T0*
_output_shapes

:L
add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Y
add_9AddV2truediv_13:z:0add_9/y:output:0*
T0*
_output_shapes

:@
Log_3Log	add_9:z:0*
T0*
_output_shapes

:Q
truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_14RealDiv	Log_3:y:0truediv_14/y:output:0*
T0*
_output_shapes

:I
Round_3Roundtruediv_14:z:0*
T0*
_output_shapes

:L
Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_4PowPow_4/x:output:0Round_3:y:0*
T0*
_output_shapes

:B
Abs_4Abstruediv:z:0*
T0*
_output_shapes

:@T

truediv_15RealDiv	Abs_4:y:0	Pow_4:z:0*
T0*
_output_shapes

:@M
add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_10AddV2truediv_15:z:0add_10/y:output:0*
T0*
_output_shapes

:@E
Floor_3Floor
add_10:z:0*
T0*
_output_shapes

:@M
Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_3LessFloor_3:y:0Less_3/y:output:0*
T0*
_output_shapes

:@D
Sign_3Signtruediv:z:0*
T0*
_output_shapes

:@r
!ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      V
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_3Fill*ones_like_3/Shape/shape_as_tensor:output:0ones_like_3/Const:output:0*
T0*
_output_shapes

:@M
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A_
mul_13Mulones_like_3:output:0mul_13/y:output:0*
T0*
_output_shapes

:@Q
truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @a

truediv_16RealDiv
mul_13:z:0truediv_16/y:output:0*
T0*
_output_shapes

:@h

SelectV2_3SelectV2
Less_3:z:0Floor_3:y:0truediv_16:z:0*
T0*
_output_shapes

:@W
mul_14Mul
Sign_3:y:0SelectV2_3:output:0*
T0*
_output_shapes

:@O
Mul_15Multruediv:z:0
mul_14:z:0*
T0*
_output_shapes

:@b
Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_6Mean
Mul_15:z:0!Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(N
Mul_16Mul
mul_14:z:0
mul_14:z:0*
T0*
_output_shapes

:@b
Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_7Mean
Mul_16:z:0!Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(M
add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3\
add_11AddV2Mean_7:output:0add_11/y:output:0*
T0*
_output_shapes

:[

truediv_17RealDivMean_6:output:0
add_11:z:0*
T0*
_output_shapes

:M
add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3[
add_12AddV2truediv_17:z:0add_12/y:output:0*
T0*
_output_shapes

:A
Log_4Log
add_12:z:0*
T0*
_output_shapes

:Q
truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_18RealDiv	Log_4:y:0truediv_18/y:output:0*
T0*
_output_shapes

:I
Round_4Roundtruediv_18:z:0*
T0*
_output_shapes

:L
Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_5PowPow_5/x:output:0Round_4:y:0*
T0*
_output_shapes

:B
Abs_5Abstruediv:z:0*
T0*
_output_shapes

:@T

truediv_19RealDiv	Abs_5:y:0	Pow_5:z:0*
T0*
_output_shapes

:@M
add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_13AddV2truediv_19:z:0add_13/y:output:0*
T0*
_output_shapes

:@E
Floor_4Floor
add_13:z:0*
T0*
_output_shapes

:@M
Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_4LessFloor_4:y:0Less_4/y:output:0*
T0*
_output_shapes

:@D
Sign_4Signtruediv:z:0*
T0*
_output_shapes

:@r
!ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@      V
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_4Fill*ones_like_4/Shape/shape_as_tensor:output:0ones_like_4/Const:output:0*
T0*
_output_shapes

:@M
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A_
mul_17Mulones_like_4:output:0mul_17/y:output:0*
T0*
_output_shapes

:@Q
truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @a

truediv_20RealDiv
mul_17:z:0truediv_20/y:output:0*
T0*
_output_shapes

:@h

SelectV2_4SelectV2
Less_4:z:0Floor_4:y:0truediv_20:z:0*
T0*
_output_shapes

:@W
mul_18Mul
Sign_4:y:0SelectV2_4:output:0*
T0*
_output_shapes

:@O
Mul_19Multruediv:z:0
mul_18:z:0*
T0*
_output_shapes

:@b
Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_8Mean
Mul_19:z:0!Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(N
Mul_20Mul
mul_18:z:0
mul_18:z:0*
T0*
_output_shapes

:@b
Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_9Mean
Mul_20:z:0!Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(M
add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3\
add_14AddV2Mean_9:output:0add_14/y:output:0*
T0*
_output_shapes

:[

truediv_21RealDivMean_8:output:0
add_14:z:0*
T0*
_output_shapes

:M
add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3[
add_15AddV2truediv_21:z:0add_15/y:output:0*
T0*
_output_shapes

:A
Log_5Log
add_15:z:0*
T0*
_output_shapes

:Q
truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_22RealDiv	Log_5:y:0truediv_22/y:output:0*
T0*
_output_shapes

:I
Round_5Roundtruediv_22:z:0*
T0*
_output_shapes

:L
Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_6PowPow_6/x:output:0Round_5:y:0*
T0*
_output_shapes

:M
mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AT
mul_21Mul	Pow_6:z:0mul_21/y:output:0*
T0*
_output_shapes

:M
mul_22MulCast:y:0truediv:z:0*
T0*
_output_shapes

:@L
mul_23MulCast:y:0
mul_18:z:0*
T0*
_output_shapes

:@Q
truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aa

truediv_23RealDiv
mul_23:z:0truediv_23/y:output:0*
T0*
_output_shapes

:@R
mul_24Mul
mul_21:z:0truediv_23:z:0*
T0*
_output_shapes

:@?
NegNeg
mul_22:z:0*
T0*
_output_shapes

:@M
add_16AddV2Neg:y:0
mul_24:z:0*
T0*
_output_shapes

:@M
mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mul_25Mulmul_25/x:output:0
add_16:z:0*
T0*
_output_shapes

:@Q
StopGradientStopGradient
mul_25:z:0*
T0*
_output_shapes

:@[
add_17AddV2
mul_22:z:0StopGradient:output:0*
T0*
_output_shapes

:@V
MatMulMatMulinputs
add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_7PowPow_7/x:output:0Pow_7/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0M
mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A_
mul_26MulReadVariableOp_1:value:0mul_26/y:output:0*
T0*
_output_shapes
:R

truediv_24RealDiv
mul_26:z:0
Cast_1:y:0*
T0*
_output_shapes
:A
Neg_1Negtruediv_24:z:0*
T0*
_output_shapes
:E
Round_6Roundtruediv_24:z:0*
T0*
_output_shapes
:L
add_18AddV2	Neg_1:y:0Round_6:y:0*
T0*
_output_shapes
:O
StopGradient_1StopGradient
add_18:z:0*
T0*
_output_shapes
:]
add_19AddV2truediv_24:z:0StopGradient_1:output:0*
T0*
_output_shapes
:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@s
clip_by_value/MinimumMinimum
add_19:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ár
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:Q
mul_27Mul
Cast_1:y:0clip_by_value:z:0*
T0*
_output_shapes
:Q
truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A]

truediv_25RealDiv
mul_27:z:0truediv_25/y:output:0*
T0*
_output_shapes
:M
mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mul_28Mulmul_28/x:output:0truediv_25:z:0*
T0*
_output_shapes
:f
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0K
Neg_2NegReadVariableOp_2:value:0*
T0*
_output_shapes
:K
add_20AddV2	Neg_2:y:0
mul_28:z:0*
T0*
_output_shapes
:M
mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Q
mul_29Mulmul_29/x:output:0
add_20:z:0*
T0*
_output_shapes
:O
StopGradient_2StopGradient
mul_29:z:0*
T0*
_output_shapes
:f
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0g
add_21AddV2ReadVariableOp_3:value:0StopGradient_2:output:0*
T0*
_output_shapes
:b
BiasAddBiasAddMatMul:product:0
add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_3:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
è¨
ÿ
B__inference_dense_1_layer_call_and_return_conditional_losses_79900

inputs)
readvariableop_resource:@@'
readvariableop_1_resource:@
identity¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:@@*
dtype0]
truedivRealDivReadVariableOp:value:0Cast:y:0*
T0*
_output_shapes

:@@@
AbsAbstruediv:z:0*
T0*
_output_shapes

:@@_
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: m
MaxMaxAbs:y:0Max/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Q
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes

:@P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A\
	truediv_1RealDivmul:z:0truediv_1/y:output:0*
T0*
_output_shapes

:@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3T
addAddV2truediv_1:z:0add/y:output:0*
T0*
_output_shapes

:@<
LogLogadd:z:0*
T0*
_output_shapes

:@P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?\
	truediv_2RealDivLog:y:0truediv_2/y:output:0*
T0*
_output_shapes

:@F
RoundRoundtruediv_2:z:0*
T0*
_output_shapes

:@L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @R
Pow_1PowPow_1/x:output:0	Round:y:0*
T0*
_output_shapes

:@B
Abs_1Abstruediv:z:0*
T0*
_output_shapes

:@@S
	truediv_3RealDiv	Abs_1:y:0	Pow_1:z:0*
T0*
_output_shapes

:@@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
add_1AddV2truediv_3:z:0add_1/y:output:0*
T0*
_output_shapes

:@@B
FloorFloor	add_1:z:0*
T0*
_output_shapes

:@@K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@Q
LessLess	Floor:y:0Less/y:output:0*
T0*
_output_shapes

:@@B
SignSigntruediv:z:0*
T0*
_output_shapes

:@@p
ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
	ones_likeFill(ones_like/Shape/shape_as_tensor:output:0ones_like/Const:output:0*
T0*
_output_shapes

:@@L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A[
mul_1Mulones_like:output:0mul_1/y:output:0*
T0*
_output_shapes

:@@P
truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
	truediv_4RealDiv	mul_1:z:0truediv_4/y:output:0*
T0*
_output_shapes

:@@a
SelectV2SelectV2Less:z:0	Floor:y:0truediv_4:z:0*
T0*
_output_shapes

:@@R
mul_2MulSign:y:0SelectV2:output:0*
T0*
_output_shapes

:@@M
Mul_3Multruediv:z:0	mul_2:z:0*
T0*
_output_shapes

:@@`
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: r
MeanMean	Mul_3:z:0Mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(K
Mul_4Mul	mul_2:z:0	mul_2:z:0*
T0*
_output_shapes

:@@b
Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_1Mean	Mul_4:z:0!Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_2AddV2Mean_1:output:0add_2/y:output:0*
T0*
_output_shapes

:@W
	truediv_5RealDivMean:output:0	add_2:z:0*
T0*
_output_shapes

:@L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3X
add_3AddV2truediv_5:z:0add_3/y:output:0*
T0*
_output_shapes

:@@
Log_1Log	add_3:z:0*
T0*
_output_shapes

:@P
truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?^
	truediv_6RealDiv	Log_1:y:0truediv_6/y:output:0*
T0*
_output_shapes

:@H
Round_1Roundtruediv_6:z:0*
T0*
_output_shapes

:@L
Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_2PowPow_2/x:output:0Round_1:y:0*
T0*
_output_shapes

:@B
Abs_2Abstruediv:z:0*
T0*
_output_shapes

:@@S
	truediv_7RealDiv	Abs_2:y:0	Pow_2:z:0*
T0*
_output_shapes

:@@L
add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
add_4AddV2truediv_7:z:0add_4/y:output:0*
T0*
_output_shapes

:@@D
Floor_1Floor	add_4:z:0*
T0*
_output_shapes

:@@M
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_1LessFloor_1:y:0Less_1/y:output:0*
T0*
_output_shapes

:@@D
Sign_1Signtruediv:z:0*
T0*
_output_shapes

:@@r
!ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_1Fill*ones_like_1/Shape/shape_as_tensor:output:0ones_like_1/Const:output:0*
T0*
_output_shapes

:@@L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A]
mul_5Mulones_like_1:output:0mul_5/y:output:0*
T0*
_output_shapes

:@@P
truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
	truediv_8RealDiv	mul_5:z:0truediv_8/y:output:0*
T0*
_output_shapes

:@@g

SelectV2_1SelectV2
Less_1:z:0Floor_1:y:0truediv_8:z:0*
T0*
_output_shapes

:@@V
mul_6Mul
Sign_1:y:0SelectV2_1:output:0*
T0*
_output_shapes

:@@M
Mul_7Multruediv:z:0	mul_6:z:0*
T0*
_output_shapes

:@@b
Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_2Mean	Mul_7:z:0!Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(K
Mul_8Mul	mul_6:z:0	mul_6:z:0*
T0*
_output_shapes

:@@b
Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_3Mean	Mul_8:z:0!Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_5AddV2Mean_3:output:0add_5/y:output:0*
T0*
_output_shapes

:@Y
	truediv_9RealDivMean_2:output:0	add_5:z:0*
T0*
_output_shapes

:@L
add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3X
add_6AddV2truediv_9:z:0add_6/y:output:0*
T0*
_output_shapes

:@@
Log_2Log	add_6:z:0*
T0*
_output_shapes

:@Q
truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_10RealDiv	Log_2:y:0truediv_10/y:output:0*
T0*
_output_shapes

:@I
Round_2Roundtruediv_10:z:0*
T0*
_output_shapes

:@L
Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_3PowPow_3/x:output:0Round_2:y:0*
T0*
_output_shapes

:@B
Abs_3Abstruediv:z:0*
T0*
_output_shapes

:@@T

truediv_11RealDiv	Abs_3:y:0	Pow_3:z:0*
T0*
_output_shapes

:@@L
add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Y
add_7AddV2truediv_11:z:0add_7/y:output:0*
T0*
_output_shapes

:@@D
Floor_2Floor	add_7:z:0*
T0*
_output_shapes

:@@M
Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_2LessFloor_2:y:0Less_2/y:output:0*
T0*
_output_shapes

:@@D
Sign_2Signtruediv:z:0*
T0*
_output_shapes

:@@r
!ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_2Fill*ones_like_2/Shape/shape_as_tensor:output:0ones_like_2/Const:output:0*
T0*
_output_shapes

:@@L
mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A]
mul_9Mulones_like_2:output:0mul_9/y:output:0*
T0*
_output_shapes

:@@Q
truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @`

truediv_12RealDiv	mul_9:z:0truediv_12/y:output:0*
T0*
_output_shapes

:@@h

SelectV2_2SelectV2
Less_2:z:0Floor_2:y:0truediv_12:z:0*
T0*
_output_shapes

:@@W
mul_10Mul
Sign_2:y:0SelectV2_2:output:0*
T0*
_output_shapes

:@@O
Mul_11Multruediv:z:0
mul_10:z:0*
T0*
_output_shapes

:@@b
Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_4Mean
Mul_11:z:0!Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_12Mul
mul_10:z:0
mul_10:z:0*
T0*
_output_shapes

:@@b
Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_5Mean
Mul_12:z:0!Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_8AddV2Mean_5:output:0add_8/y:output:0*
T0*
_output_shapes

:@Z

truediv_13RealDivMean_4:output:0	add_8:z:0*
T0*
_output_shapes

:@L
add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Y
add_9AddV2truediv_13:z:0add_9/y:output:0*
T0*
_output_shapes

:@@
Log_3Log	add_9:z:0*
T0*
_output_shapes

:@Q
truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_14RealDiv	Log_3:y:0truediv_14/y:output:0*
T0*
_output_shapes

:@I
Round_3Roundtruediv_14:z:0*
T0*
_output_shapes

:@L
Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_4PowPow_4/x:output:0Round_3:y:0*
T0*
_output_shapes

:@B
Abs_4Abstruediv:z:0*
T0*
_output_shapes

:@@T

truediv_15RealDiv	Abs_4:y:0	Pow_4:z:0*
T0*
_output_shapes

:@@M
add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_10AddV2truediv_15:z:0add_10/y:output:0*
T0*
_output_shapes

:@@E
Floor_3Floor
add_10:z:0*
T0*
_output_shapes

:@@M
Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_3LessFloor_3:y:0Less_3/y:output:0*
T0*
_output_shapes

:@@D
Sign_3Signtruediv:z:0*
T0*
_output_shapes

:@@r
!ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   V
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_3Fill*ones_like_3/Shape/shape_as_tensor:output:0ones_like_3/Const:output:0*
T0*
_output_shapes

:@@M
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A_
mul_13Mulones_like_3:output:0mul_13/y:output:0*
T0*
_output_shapes

:@@Q
truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @a

truediv_16RealDiv
mul_13:z:0truediv_16/y:output:0*
T0*
_output_shapes

:@@h

SelectV2_3SelectV2
Less_3:z:0Floor_3:y:0truediv_16:z:0*
T0*
_output_shapes

:@@W
mul_14Mul
Sign_3:y:0SelectV2_3:output:0*
T0*
_output_shapes

:@@O
Mul_15Multruediv:z:0
mul_14:z:0*
T0*
_output_shapes

:@@b
Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_6Mean
Mul_15:z:0!Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_16Mul
mul_14:z:0
mul_14:z:0*
T0*
_output_shapes

:@@b
Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_7Mean
Mul_16:z:0!Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(M
add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3\
add_11AddV2Mean_7:output:0add_11/y:output:0*
T0*
_output_shapes

:@[

truediv_17RealDivMean_6:output:0
add_11:z:0*
T0*
_output_shapes

:@M
add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3[
add_12AddV2truediv_17:z:0add_12/y:output:0*
T0*
_output_shapes

:@A
Log_4Log
add_12:z:0*
T0*
_output_shapes

:@Q
truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_18RealDiv	Log_4:y:0truediv_18/y:output:0*
T0*
_output_shapes

:@I
Round_4Roundtruediv_18:z:0*
T0*
_output_shapes

:@L
Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_5PowPow_5/x:output:0Round_4:y:0*
T0*
_output_shapes

:@B
Abs_5Abstruediv:z:0*
T0*
_output_shapes

:@@T

truediv_19RealDiv	Abs_5:y:0	Pow_5:z:0*
T0*
_output_shapes

:@@M
add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_13AddV2truediv_19:z:0add_13/y:output:0*
T0*
_output_shapes

:@@E
Floor_4Floor
add_13:z:0*
T0*
_output_shapes

:@@M
Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_4LessFloor_4:y:0Less_4/y:output:0*
T0*
_output_shapes

:@@D
Sign_4Signtruediv:z:0*
T0*
_output_shapes

:@@r
!ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@   @   V
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_4Fill*ones_like_4/Shape/shape_as_tensor:output:0ones_like_4/Const:output:0*
T0*
_output_shapes

:@@M
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A_
mul_17Mulones_like_4:output:0mul_17/y:output:0*
T0*
_output_shapes

:@@Q
truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @a

truediv_20RealDiv
mul_17:z:0truediv_20/y:output:0*
T0*
_output_shapes

:@@h

SelectV2_4SelectV2
Less_4:z:0Floor_4:y:0truediv_20:z:0*
T0*
_output_shapes

:@@W
mul_18Mul
Sign_4:y:0SelectV2_4:output:0*
T0*
_output_shapes

:@@O
Mul_19Multruediv:z:0
mul_18:z:0*
T0*
_output_shapes

:@@b
Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_8Mean
Mul_19:z:0!Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_20Mul
mul_18:z:0
mul_18:z:0*
T0*
_output_shapes

:@@b
Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_9Mean
Mul_20:z:0!Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(M
add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3\
add_14AddV2Mean_9:output:0add_14/y:output:0*
T0*
_output_shapes

:@[

truediv_21RealDivMean_8:output:0
add_14:z:0*
T0*
_output_shapes

:@M
add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3[
add_15AddV2truediv_21:z:0add_15/y:output:0*
T0*
_output_shapes

:@A
Log_5Log
add_15:z:0*
T0*
_output_shapes

:@Q
truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_22RealDiv	Log_5:y:0truediv_22/y:output:0*
T0*
_output_shapes

:@I
Round_5Roundtruediv_22:z:0*
T0*
_output_shapes

:@L
Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_6PowPow_6/x:output:0Round_5:y:0*
T0*
_output_shapes

:@M
mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AT
mul_21Mul	Pow_6:z:0mul_21/y:output:0*
T0*
_output_shapes

:@M
mul_22MulCast:y:0truediv:z:0*
T0*
_output_shapes

:@@L
mul_23MulCast:y:0
mul_18:z:0*
T0*
_output_shapes

:@@Q
truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aa

truediv_23RealDiv
mul_23:z:0truediv_23/y:output:0*
T0*
_output_shapes

:@@R
mul_24Mul
mul_21:z:0truediv_23:z:0*
T0*
_output_shapes

:@@?
NegNeg
mul_22:z:0*
T0*
_output_shapes

:@@M
add_16AddV2Neg:y:0
mul_24:z:0*
T0*
_output_shapes

:@@M
mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mul_25Mulmul_25/x:output:0
add_16:z:0*
T0*
_output_shapes

:@@Q
StopGradientStopGradient
mul_25:z:0*
T0*
_output_shapes

:@@[
add_17AddV2
mul_22:z:0StopGradient:output:0*
T0*
_output_shapes

:@@V
MatMulMatMulinputs
add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@I
Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_7PowPow_7/x:output:0Pow_7/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0M
mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A_
mul_26MulReadVariableOp_1:value:0mul_26/y:output:0*
T0*
_output_shapes
:@R

truediv_24RealDiv
mul_26:z:0
Cast_1:y:0*
T0*
_output_shapes
:@A
Neg_1Negtruediv_24:z:0*
T0*
_output_shapes
:@E
Round_6Roundtruediv_24:z:0*
T0*
_output_shapes
:@L
add_18AddV2	Neg_1:y:0Round_6:y:0*
T0*
_output_shapes
:@O
StopGradient_1StopGradient
add_18:z:0*
T0*
_output_shapes
:@]
add_19AddV2truediv_24:z:0StopGradient_1:output:0*
T0*
_output_shapes
:@\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@s
clip_by_value/MinimumMinimum
add_19:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:@T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ár
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:@Q
mul_27Mul
Cast_1:y:0clip_by_value:z:0*
T0*
_output_shapes
:@Q
truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A]

truediv_25RealDiv
mul_27:z:0truediv_25/y:output:0*
T0*
_output_shapes
:@M
mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mul_28Mulmul_28/x:output:0truediv_25:z:0*
T0*
_output_shapes
:@f
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0K
Neg_2NegReadVariableOp_2:value:0*
T0*
_output_shapes
:@K
add_20AddV2	Neg_2:y:0
mul_28:z:0*
T0*
_output_shapes
:@M
mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Q
mul_29Mulmul_29/x:output:0
add_20:z:0*
T0*
_output_shapes
:@O
StopGradient_2StopGradient
mul_29:z:0*
T0*
_output_shapes
:@f
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0g
add_21AddV2ReadVariableOp_3:value:0StopGradient_2:output:0*
T0*
_output_shapes
:@b
BiasAddBiasAddMatMul:product:0
add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_3:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

C
'__inference_re_lu_1_layer_call_fn_80012

inputs
identity­
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_75969`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ÿ:
ì
H__inference_my_sequential_layer_call_and_return_conditional_losses_76921
input_1
dense_76867:v@
dense_76869:@'
batch_normalization_76872:@'
batch_normalization_76874:@'
batch_normalization_76876:@'
batch_normalization_76878:@
dense_1_76883:@@
dense_1_76885:@)
batch_normalization_1_76888:@)
batch_normalization_1_76890:@)
batch_normalization_1_76892:@)
batch_normalization_1_76894:@
dense_2_76899:@@
dense_2_76901:@)
batch_normalization_2_76904:@)
batch_normalization_2_76906:@)
batch_normalization_2_76908:@)
batch_normalization_2_76910:@
dense_3_76915:@
dense_3_76917:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCallâ
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_76867dense_76869*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75649ó
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_76872batch_normalization_76874batch_normalization_76876batch_normalization_76878*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_75178ã
dropout/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_75669Ë
re_lu/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_75687
dense_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0dense_1_76883dense_1_76885*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75931
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_76888batch_normalization_1_76890batch_normalization_1_76892batch_normalization_1_76894*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_75260é
dropout_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_75951Ñ
re_lu_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_75969
dense_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0dense_2_76899dense_2_76901*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_76213
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_76904batch_normalization_2_76906batch_normalization_2_76908batch_normalization_2_76910*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_75342é
dropout_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_76233Ñ
re_lu_2/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_76251
dense_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0dense_3_76915dense_3_76917*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_76495w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿv: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!
_user_specified_name	input_1
Í
¯
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_75342

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ð	
a
B__inference_dropout_layer_call_and_return_conditional_losses_76659

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
µ
ò
-__inference_my_sequential_layer_call_fn_76864
input_1
unknown:v@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identity¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_my_sequential_layer_call_and_return_conditional_losses_76776o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿv: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!
_user_specified_name	input_1
¸
ñ
-__inference_my_sequential_layer_call_fn_77076

inputs
unknown:v@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identity¢StatefulPartitionedCallÐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_my_sequential_layer_call_and_return_conditional_losses_76502o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿv: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_nameinputs
§å
$
!__inference__traced_restore_81033
file_prefix/
assignvariableop_dense_kernel:v@+
assignvariableop_1_dense_bias:@:
,assignvariableop_2_batch_normalization_gamma:@9
+assignvariableop_3_batch_normalization_beta:@@
2assignvariableop_4_batch_normalization_moving_mean:@D
6assignvariableop_5_batch_normalization_moving_variance:@3
!assignvariableop_6_dense_1_kernel:@@-
assignvariableop_7_dense_1_bias:@<
.assignvariableop_8_batch_normalization_1_gamma:@;
-assignvariableop_9_batch_normalization_1_beta:@C
5assignvariableop_10_batch_normalization_1_moving_mean:@G
9assignvariableop_11_batch_normalization_1_moving_variance:@4
"assignvariableop_12_dense_2_kernel:@@.
 assignvariableop_13_dense_2_bias:@=
/assignvariableop_14_batch_normalization_2_gamma:@<
.assignvariableop_15_batch_normalization_2_beta:@C
5assignvariableop_16_batch_normalization_2_moving_mean:@G
9assignvariableop_17_batch_normalization_2_moving_variance:@4
"assignvariableop_18_dense_3_kernel:@.
 assignvariableop_19_dense_3_bias:'
assignvariableop_20_adam_iter:	 )
assignvariableop_21_adam_beta_1: )
assignvariableop_22_adam_beta_2: (
assignvariableop_23_adam_decay: 0
&assignvariableop_24_adam_learning_rate: %
assignvariableop_25_total_1: %
assignvariableop_26_count_1: #
assignvariableop_27_total: #
assignvariableop_28_count: 9
'assignvariableop_29_adam_dense_kernel_m:v@3
%assignvariableop_30_adam_dense_bias_m:@B
4assignvariableop_31_adam_batch_normalization_gamma_m:@A
3assignvariableop_32_adam_batch_normalization_beta_m:@;
)assignvariableop_33_adam_dense_1_kernel_m:@@5
'assignvariableop_34_adam_dense_1_bias_m:@D
6assignvariableop_35_adam_batch_normalization_1_gamma_m:@C
5assignvariableop_36_adam_batch_normalization_1_beta_m:@;
)assignvariableop_37_adam_dense_2_kernel_m:@@5
'assignvariableop_38_adam_dense_2_bias_m:@D
6assignvariableop_39_adam_batch_normalization_2_gamma_m:@C
5assignvariableop_40_adam_batch_normalization_2_beta_m:@;
)assignvariableop_41_adam_dense_3_kernel_m:@5
'assignvariableop_42_adam_dense_3_bias_m:9
'assignvariableop_43_adam_dense_kernel_v:v@3
%assignvariableop_44_adam_dense_bias_v:@B
4assignvariableop_45_adam_batch_normalization_gamma_v:@A
3assignvariableop_46_adam_batch_normalization_beta_v:@;
)assignvariableop_47_adam_dense_1_kernel_v:@@5
'assignvariableop_48_adam_dense_1_bias_v:@D
6assignvariableop_49_adam_batch_normalization_1_gamma_v:@C
5assignvariableop_50_adam_batch_normalization_1_beta_v:@;
)assignvariableop_51_adam_dense_2_kernel_v:@@5
'assignvariableop_52_adam_dense_2_bias_v:@D
6assignvariableop_53_adam_batch_normalization_2_gamma_v:@C
5assignvariableop_54_adam_batch_normalization_2_beta_v:@;
)assignvariableop_55_adam_dense_3_kernel_v:@5
'assignvariableop_56_adam_dense_3_bias_v:
identity_58¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ñ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*÷
valueíBê:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHå
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ã
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*þ
_output_shapesë
è::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_dense_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_batch_normalization_gamma_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_32AssignVariableOp3assignvariableop_32_adam_batch_normalization_beta_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_batch_normalization_1_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_batch_normalization_1_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_2_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_2_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_batch_normalization_2_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_40AssignVariableOp5assignvariableop_40_adam_batch_normalization_2_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_3_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_3_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_dense_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp%assignvariableop_44_adam_dense_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_45AssignVariableOp4assignvariableop_45_adam_batch_normalization_gamma_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_46AssignVariableOp3assignvariableop_46_adam_batch_normalization_beta_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_1_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_1_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_batch_normalization_1_gamma_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_batch_normalization_1_beta_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_2_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_dense_2_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_batch_normalization_2_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_batch_normalization_2_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_dense_3_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_dense_3_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 µ

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: ¢

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_58Identity_58:output:0*
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¾

\
@__inference_re_lu_layer_call_and_return_conditional_losses_79649

inputs
identityF
SignSigninputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
AbsAbsSign:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
subSubsub/x:output:0Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
addAddV2Sign:y:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
TanhTanhinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
NegNegTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mulMulmul/x:output:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
add_1AddV2Neg:y:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
StopGradientStopGradient	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
add_2AddV2Tanh:y:0StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æ¨
ý
@__inference_dense_layer_call_and_return_conditional_losses_79521

inputs)
readvariableop_resource:v@'
readvariableop_1_resource:@
identity¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:v@*
dtype0]
truedivRealDivReadVariableOp:value:0Cast:y:0*
T0*
_output_shapes

:v@@
AbsAbstruediv:z:0*
T0*
_output_shapes

:v@_
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: m
MaxMaxAbs:y:0Max/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Q
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes

:@P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A\
	truediv_1RealDivmul:z:0truediv_1/y:output:0*
T0*
_output_shapes

:@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3T
addAddV2truediv_1:z:0add/y:output:0*
T0*
_output_shapes

:@<
LogLogadd:z:0*
T0*
_output_shapes

:@P
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?\
	truediv_2RealDivLog:y:0truediv_2/y:output:0*
T0*
_output_shapes

:@F
RoundRoundtruediv_2:z:0*
T0*
_output_shapes

:@L
Pow_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @R
Pow_1PowPow_1/x:output:0	Round:y:0*
T0*
_output_shapes

:@B
Abs_1Abstruediv:z:0*
T0*
_output_shapes

:v@S
	truediv_3RealDiv	Abs_1:y:0	Pow_1:z:0*
T0*
_output_shapes

:v@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
add_1AddV2truediv_3:z:0add_1/y:output:0*
T0*
_output_shapes

:v@B
FloorFloor	add_1:z:0*
T0*
_output_shapes

:v@K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@Q
LessLess	Floor:y:0Less/y:output:0*
T0*
_output_shapes

:v@B
SignSigntruediv:z:0*
T0*
_output_shapes

:v@p
ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
	ones_likeFill(ones_like/Shape/shape_as_tensor:output:0ones_like/Const:output:0*
T0*
_output_shapes

:v@L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A[
mul_1Mulones_like:output:0mul_1/y:output:0*
T0*
_output_shapes

:v@P
truediv_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
	truediv_4RealDiv	mul_1:z:0truediv_4/y:output:0*
T0*
_output_shapes

:v@a
SelectV2SelectV2Less:z:0	Floor:y:0truediv_4:z:0*
T0*
_output_shapes

:v@R
mul_2MulSign:y:0SelectV2:output:0*
T0*
_output_shapes

:v@M
Mul_3Multruediv:z:0	mul_2:z:0*
T0*
_output_shapes

:v@`
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: r
MeanMean	Mul_3:z:0Mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(K
Mul_4Mul	mul_2:z:0	mul_2:z:0*
T0*
_output_shapes

:v@b
Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_1Mean	Mul_4:z:0!Mean_1/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_2AddV2Mean_1:output:0add_2/y:output:0*
T0*
_output_shapes

:@W
	truediv_5RealDivMean:output:0	add_2:z:0*
T0*
_output_shapes

:@L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3X
add_3AddV2truediv_5:z:0add_3/y:output:0*
T0*
_output_shapes

:@@
Log_1Log	add_3:z:0*
T0*
_output_shapes

:@P
truediv_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?^
	truediv_6RealDiv	Log_1:y:0truediv_6/y:output:0*
T0*
_output_shapes

:@H
Round_1Roundtruediv_6:z:0*
T0*
_output_shapes

:@L
Pow_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_2PowPow_2/x:output:0Round_1:y:0*
T0*
_output_shapes

:@B
Abs_2Abstruediv:z:0*
T0*
_output_shapes

:v@S
	truediv_7RealDiv	Abs_2:y:0	Pow_2:z:0*
T0*
_output_shapes

:v@L
add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
add_4AddV2truediv_7:z:0add_4/y:output:0*
T0*
_output_shapes

:v@D
Floor_1Floor	add_4:z:0*
T0*
_output_shapes

:v@M
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_1LessFloor_1:y:0Less_1/y:output:0*
T0*
_output_shapes

:v@D
Sign_1Signtruediv:z:0*
T0*
_output_shapes

:v@r
!ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_1Fill*ones_like_1/Shape/shape_as_tensor:output:0ones_like_1/Const:output:0*
T0*
_output_shapes

:v@L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A]
mul_5Mulones_like_1:output:0mul_5/y:output:0*
T0*
_output_shapes

:v@P
truediv_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
	truediv_8RealDiv	mul_5:z:0truediv_8/y:output:0*
T0*
_output_shapes

:v@g

SelectV2_1SelectV2
Less_1:z:0Floor_1:y:0truediv_8:z:0*
T0*
_output_shapes

:v@V
mul_6Mul
Sign_1:y:0SelectV2_1:output:0*
T0*
_output_shapes

:v@M
Mul_7Multruediv:z:0	mul_6:z:0*
T0*
_output_shapes

:v@b
Mean_2/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_2Mean	Mul_7:z:0!Mean_2/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(K
Mul_8Mul	mul_6:z:0	mul_6:z:0*
T0*
_output_shapes

:v@b
Mean_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: v
Mean_3Mean	Mul_8:z:0!Mean_3/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_5AddV2Mean_3:output:0add_5/y:output:0*
T0*
_output_shapes

:@Y
	truediv_9RealDivMean_2:output:0	add_5:z:0*
T0*
_output_shapes

:@L
add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3X
add_6AddV2truediv_9:z:0add_6/y:output:0*
T0*
_output_shapes

:@@
Log_2Log	add_6:z:0*
T0*
_output_shapes

:@Q
truediv_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_10RealDiv	Log_2:y:0truediv_10/y:output:0*
T0*
_output_shapes

:@I
Round_2Roundtruediv_10:z:0*
T0*
_output_shapes

:@L
Pow_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_3PowPow_3/x:output:0Round_2:y:0*
T0*
_output_shapes

:@B
Abs_3Abstruediv:z:0*
T0*
_output_shapes

:v@T

truediv_11RealDiv	Abs_3:y:0	Pow_3:z:0*
T0*
_output_shapes

:v@L
add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Y
add_7AddV2truediv_11:z:0add_7/y:output:0*
T0*
_output_shapes

:v@D
Floor_2Floor	add_7:z:0*
T0*
_output_shapes

:v@M
Less_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_2LessFloor_2:y:0Less_2/y:output:0*
T0*
_output_shapes

:v@D
Sign_2Signtruediv:z:0*
T0*
_output_shapes

:v@r
!ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_2Fill*ones_like_2/Shape/shape_as_tensor:output:0ones_like_2/Const:output:0*
T0*
_output_shapes

:v@L
mul_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A]
mul_9Mulones_like_2:output:0mul_9/y:output:0*
T0*
_output_shapes

:v@Q
truediv_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @`

truediv_12RealDiv	mul_9:z:0truediv_12/y:output:0*
T0*
_output_shapes

:v@h

SelectV2_2SelectV2
Less_2:z:0Floor_2:y:0truediv_12:z:0*
T0*
_output_shapes

:v@W
mul_10Mul
Sign_2:y:0SelectV2_2:output:0*
T0*
_output_shapes

:v@O
Mul_11Multruediv:z:0
mul_10:z:0*
T0*
_output_shapes

:v@b
Mean_4/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_4Mean
Mul_11:z:0!Mean_4/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_12Mul
mul_10:z:0
mul_10:z:0*
T0*
_output_shapes

:v@b
Mean_5/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_5Mean
Mul_12:z:0!Mean_5/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(L
add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Z
add_8AddV2Mean_5:output:0add_8/y:output:0*
T0*
_output_shapes

:@Z

truediv_13RealDivMean_4:output:0	add_8:z:0*
T0*
_output_shapes

:@L
add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3Y
add_9AddV2truediv_13:z:0add_9/y:output:0*
T0*
_output_shapes

:@@
Log_3Log	add_9:z:0*
T0*
_output_shapes

:@Q
truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_14RealDiv	Log_3:y:0truediv_14/y:output:0*
T0*
_output_shapes

:@I
Round_3Roundtruediv_14:z:0*
T0*
_output_shapes

:@L
Pow_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_4PowPow_4/x:output:0Round_3:y:0*
T0*
_output_shapes

:@B
Abs_4Abstruediv:z:0*
T0*
_output_shapes

:v@T

truediv_15RealDiv	Abs_4:y:0	Pow_4:z:0*
T0*
_output_shapes

:v@M
add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_10AddV2truediv_15:z:0add_10/y:output:0*
T0*
_output_shapes

:v@E
Floor_3Floor
add_10:z:0*
T0*
_output_shapes

:v@M
Less_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_3LessFloor_3:y:0Less_3/y:output:0*
T0*
_output_shapes

:v@D
Sign_3Signtruediv:z:0*
T0*
_output_shapes

:v@r
!ones_like_3/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   V
ones_like_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_3Fill*ones_like_3/Shape/shape_as_tensor:output:0ones_like_3/Const:output:0*
T0*
_output_shapes

:v@M
mul_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A_
mul_13Mulones_like_3:output:0mul_13/y:output:0*
T0*
_output_shapes

:v@Q
truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @a

truediv_16RealDiv
mul_13:z:0truediv_16/y:output:0*
T0*
_output_shapes

:v@h

SelectV2_3SelectV2
Less_3:z:0Floor_3:y:0truediv_16:z:0*
T0*
_output_shapes

:v@W
mul_14Mul
Sign_3:y:0SelectV2_3:output:0*
T0*
_output_shapes

:v@O
Mul_15Multruediv:z:0
mul_14:z:0*
T0*
_output_shapes

:v@b
Mean_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_6Mean
Mul_15:z:0!Mean_6/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_16Mul
mul_14:z:0
mul_14:z:0*
T0*
_output_shapes

:v@b
Mean_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_7Mean
Mul_16:z:0!Mean_7/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(M
add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3\
add_11AddV2Mean_7:output:0add_11/y:output:0*
T0*
_output_shapes

:@[

truediv_17RealDivMean_6:output:0
add_11:z:0*
T0*
_output_shapes

:@M
add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3[
add_12AddV2truediv_17:z:0add_12/y:output:0*
T0*
_output_shapes

:@A
Log_4Log
add_12:z:0*
T0*
_output_shapes

:@Q
truediv_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_18RealDiv	Log_4:y:0truediv_18/y:output:0*
T0*
_output_shapes

:@I
Round_4Roundtruediv_18:z:0*
T0*
_output_shapes

:@L
Pow_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_5PowPow_5/x:output:0Round_4:y:0*
T0*
_output_shapes

:@B
Abs_5Abstruediv:z:0*
T0*
_output_shapes

:v@T

truediv_19RealDiv	Abs_5:y:0	Pow_5:z:0*
T0*
_output_shapes

:v@M
add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?[
add_13AddV2truediv_19:z:0add_13/y:output:0*
T0*
_output_shapes

:v@E
Floor_4Floor
add_13:z:0*
T0*
_output_shapes

:v@M
Less_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@W
Less_4LessFloor_4:y:0Less_4/y:output:0*
T0*
_output_shapes

:v@D
Sign_4Signtruediv:z:0*
T0*
_output_shapes

:v@r
!ones_like_4/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"v   @   V
ones_like_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ones_like_4Fill*ones_like_4/Shape/shape_as_tensor:output:0ones_like_4/Const:output:0*
T0*
_output_shapes

:v@M
mul_17/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A_
mul_17Mulones_like_4:output:0mul_17/y:output:0*
T0*
_output_shapes

:v@Q
truediv_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @a

truediv_20RealDiv
mul_17:z:0truediv_20/y:output:0*
T0*
_output_shapes

:v@h

SelectV2_4SelectV2
Less_4:z:0Floor_4:y:0truediv_20:z:0*
T0*
_output_shapes

:v@W
mul_18Mul
Sign_4:y:0SelectV2_4:output:0*
T0*
_output_shapes

:v@O
Mul_19Multruediv:z:0
mul_18:z:0*
T0*
_output_shapes

:v@b
Mean_8/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_8Mean
Mul_19:z:0!Mean_8/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(N
Mul_20Mul
mul_18:z:0
mul_18:z:0*
T0*
_output_shapes

:v@b
Mean_9/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: w
Mean_9Mean
Mul_20:z:0!Mean_9/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(M
add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3\
add_14AddV2Mean_9:output:0add_14/y:output:0*
T0*
_output_shapes

:@[

truediv_21RealDivMean_8:output:0
add_14:z:0*
T0*
_output_shapes

:@M
add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3[
add_15AddV2truediv_21:z:0add_15/y:output:0*
T0*
_output_shapes

:@A
Log_5Log
add_15:z:0*
T0*
_output_shapes

:@Q
truediv_22/yConst*
_output_shapes
: *
dtype0*
valueB
 *r1?`

truediv_22RealDiv	Log_5:y:0truediv_22/y:output:0*
T0*
_output_shapes

:@I
Round_5Roundtruediv_22:z:0*
T0*
_output_shapes

:@L
Pow_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
Pow_6PowPow_6/x:output:0Round_5:y:0*
T0*
_output_shapes

:@M
mul_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AT
mul_21Mul	Pow_6:z:0mul_21/y:output:0*
T0*
_output_shapes

:@M
mul_22MulCast:y:0truediv:z:0*
T0*
_output_shapes

:v@L
mul_23MulCast:y:0
mul_18:z:0*
T0*
_output_shapes

:v@Q
truediv_23/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Aa

truediv_23RealDiv
mul_23:z:0truediv_23/y:output:0*
T0*
_output_shapes

:v@R
mul_24Mul
mul_21:z:0truediv_23:z:0*
T0*
_output_shapes

:v@?
NegNeg
mul_22:z:0*
T0*
_output_shapes

:v@M
add_16AddV2Neg:y:0
mul_24:z:0*
T0*
_output_shapes

:v@M
mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mul_25Mulmul_25/x:output:0
add_16:z:0*
T0*
_output_shapes

:v@Q
StopGradientStopGradient
mul_25:z:0*
T0*
_output_shapes

:v@[
add_17AddV2
mul_22:z:0StopGradient:output:0*
T0*
_output_shapes

:v@V
MatMulMatMulinputs
add_17:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@I
Pow_7/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_7/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_7PowPow_7/x:output:0Pow_7/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_7:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0M
mul_26/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A_
mul_26MulReadVariableOp_1:value:0mul_26/y:output:0*
T0*
_output_shapes
:@R

truediv_24RealDiv
mul_26:z:0
Cast_1:y:0*
T0*
_output_shapes
:@A
Neg_1Negtruediv_24:z:0*
T0*
_output_shapes
:@E
Round_6Roundtruediv_24:z:0*
T0*
_output_shapes
:@L
add_18AddV2	Neg_1:y:0Round_6:y:0*
T0*
_output_shapes
:@O
StopGradient_1StopGradient
add_18:z:0*
T0*
_output_shapes
:@]
add_19AddV2truediv_24:z:0StopGradient_1:output:0*
T0*
_output_shapes
:@\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  à@s
clip_by_value/MinimumMinimum
add_19:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:@T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ár
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:@Q
mul_27Mul
Cast_1:y:0clip_by_value:z:0*
T0*
_output_shapes
:@Q
truediv_25/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A]

truediv_25RealDiv
mul_27:z:0truediv_25/y:output:0*
T0*
_output_shapes
:@M
mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mul_28Mulmul_28/x:output:0truediv_25:z:0*
T0*
_output_shapes
:@f
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0K
Neg_2NegReadVariableOp_2:value:0*
T0*
_output_shapes
:@K
add_20AddV2	Neg_2:y:0
mul_28:z:0*
T0*
_output_shapes
:@M
mul_29/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Q
mul_29Mulmul_29/x:output:0
add_20:z:0*
T0*
_output_shapes
:@O
StopGradient_2StopGradient
mul_29:z:0*
T0*
_output_shapes
:@f
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0g
add_21AddV2ReadVariableOp_3:value:0StopGradient_2:output:0*
T0*
_output_shapes
:@b
BiasAddBiasAddMatMul:product:0
add_21:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿv: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_3:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_nameinputs
»
ò
-__inference_my_sequential_layer_call_fn_76545
input_1
unknown:v@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_my_sequential_layer_call_and_return_conditional_losses_76502o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿv: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!
_user_specified_name	input_1
ò	
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_80386

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¦
Ð
5__inference_batch_normalization_2_layer_call_fn_80292

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_75342o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤
Ð
5__inference_batch_normalization_1_layer_call_fn_79926

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_75307o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Õ
`
B__inference_dropout_layer_call_and_return_conditional_losses_79616

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
%
ç
N__inference_batch_normalization_layer_call_and_return_conditional_losses_79601

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¬?
Õ	
H__inference_my_sequential_layer_call_and_return_conditional_losses_76776

inputs
dense_76722:v@
dense_76724:@'
batch_normalization_76727:@'
batch_normalization_76729:@'
batch_normalization_76731:@'
batch_normalization_76733:@
dense_1_76738:@@
dense_1_76740:@)
batch_normalization_1_76743:@)
batch_normalization_1_76745:@)
batch_normalization_1_76747:@)
batch_normalization_1_76749:@
dense_2_76754:@@
dense_2_76756:@)
batch_normalization_2_76759:@)
batch_normalization_2_76761:@)
batch_normalization_2_76763:@)
batch_normalization_2_76765:@
dense_3_76770:@
dense_3_76772:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCallá
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_76722dense_76724*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75649ñ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_76727batch_normalization_76729batch_normalization_76731batch_normalization_76733*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_75225ó
dropout/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_76659Ó
re_lu/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_re_lu_layer_call_and_return_conditional_losses_75687
dense_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0dense_1_76738dense_1_76740*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75931ÿ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_1_76743batch_normalization_1_76745batch_normalization_1_76747batch_normalization_1_76749*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_75307
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_76620Ù
re_lu_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_1_layer_call_and_return_conditional_losses_75969
dense_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0dense_2_76754dense_2_76756*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_76213ÿ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_2_76759batch_normalization_2_76761batch_normalization_2_76763batch_normalization_2_76765*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_75389
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_76581Ù
re_lu_2/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_re_lu_2_layer_call_and_return_conditional_losses_76251
dense_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0dense_3_76770dense_3_76772*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_76495w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿv: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_nameinputs
Í
¯
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_80325

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
À

^
B__inference_re_lu_1_layer_call_and_return_conditional_losses_75969

inputs
identityF
SignSigninputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
AbsAbsSign:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
subSubsub/x:output:0Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
addAddV2Sign:y:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
TanhTanhinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@F
NegNegTanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?U
mulMulmul/x:output:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
add_1AddV2Neg:y:0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
StopGradientStopGradient	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
add_2AddV2Tanh:y:0StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ë
­
N__inference_batch_normalization_layer_call_and_return_conditional_losses_79567

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
%
é
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_79980

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ð	
a
B__inference_dropout_layer_call_and_return_conditional_losses_79628

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Õ
`
B__inference_dropout_layer_call_and_return_conditional_losses_75669

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Í
¯
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_75260

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¾

'__inference_dense_1_layer_call_fn_79658

inputs
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75931o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
×
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_75951

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
%
é
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_80359

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¾

'__inference_dense_3_layer_call_fn_80416

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_76495o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
×
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_80374

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
×
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_79995

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ª
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿv;
dense_30
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ÄÝ
È
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
kernel_quantizer_internal
bias_quantizer_internal

quantizers

 kernel
!bias"
_tf_keras_layer
ê
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(axis
	)gamma
*beta
+moving_mean
,moving_variance"
_tf_keras_layer
¼
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3_random_generator"
_tf_keras_layer
´
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:	quantizer"
_tf_keras_layer

;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
Akernel_quantizer_internal
Bbias_quantizer_internal
C
quantizers

Dkernel
Ebias"
_tf_keras_layer
ê
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance"
_tf_keras_layer
¼
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
W_random_generator"
_tf_keras_layer
´
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^	quantizer"
_tf_keras_layer

_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
ekernel_quantizer_internal
fbias_quantizer_internal
g
quantizers

hkernel
ibias"
_tf_keras_layer
ê
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance"
_tf_keras_layer
¼
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
{_random_generator"
_tf_keras_layer
·
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	quantizer"
_tf_keras_layer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel_quantizer_internal
bias_quantizer_internal

quantizers
kernel
	bias"
_tf_keras_layer
¸
 0
!1
)2
*3
+4
,5
D6
E7
M8
N9
O10
P11
h12
i13
q14
r15
s16
t17
18
19"
trackable_list_wrapper

 0
!1
)2
*3
D4
E5
M6
N7
h8
i9
q10
r11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ñ
trace_0
trace_1
trace_2
trace_32þ
-__inference_my_sequential_layer_call_fn_76545
-__inference_my_sequential_layer_call_fn_77076
-__inference_my_sequential_layer_call_fn_77121
-__inference_my_sequential_layer_call_fn_76864¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
Ý
trace_0
trace_1
trace_2
trace_32ê
H__inference_my_sequential_layer_call_and_return_conditional_losses_78164
H__inference_my_sequential_layer_call_and_return_conditional_losses_79270
H__inference_my_sequential_layer_call_and_return_conditional_losses_76921
H__inference_my_sequential_layer_call_and_return_conditional_losses_76978¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
ËBÈ
 __inference__wrapped_model_75154input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô
	iter
beta_1
beta_2

decay
learning_rate m!m)m*mDmEmMmNmhmimqmrm	m	m  v¡!v¢)v£*v¤Dv¥Ev¦Mv§Nv¨hv©ivªqv«rv¬	v­	v®"
	optimizer
-
 serving_default"
signature_map
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ë
¦trace_02Ì
%__inference_dense_layer_call_fn_79279¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¦trace_0

§trace_02ç
@__inference_dense_layer_call_and_return_conditional_losses_79521¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z§trace_0
"
_generic_user_object
"
_generic_user_object
.
0
1"
trackable_list_wrapper
:v@2dense/kernel
:@2
dense/bias
<
)0
*1
+2
,3"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
Û
­trace_0
®trace_12 
3__inference_batch_normalization_layer_call_fn_79534
3__inference_batch_normalization_layer_call_fn_79547³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z­trace_0z®trace_1

¯trace_0
°trace_12Ö
N__inference_batch_normalization_layer_call_and_return_conditional_losses_79567
N__inference_batch_normalization_layer_call_and_return_conditional_losses_79601³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¯trace_0z°trace_1
 "
trackable_list_wrapper
':%@2batch_normalization/gamma
&:$@2batch_normalization/beta
/:-@ (2batch_normalization/moving_mean
3:1@ (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
Ã
¶trace_0
·trace_12
'__inference_dropout_layer_call_fn_79606
'__inference_dropout_layer_call_fn_79611³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¶trace_0z·trace_1
ù
¸trace_0
¹trace_12¾
B__inference_dropout_layer_call_and_return_conditional_losses_79616
B__inference_dropout_layer_call_and_return_conditional_losses_79628³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¸trace_0z¹trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ºnon_trainable_variables
»layers
¼metrics
 ½layer_regularization_losses
¾layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
ë
¿trace_02Ì
%__inference_re_lu_layer_call_fn_79633¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¿trace_0

Àtrace_02ç
@__inference_re_lu_layer_call_and_return_conditional_losses_79649¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÀtrace_0
"
_generic_user_object
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
í
Ætrace_02Î
'__inference_dense_1_layer_call_fn_79658¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÆtrace_0

Çtrace_02é
B__inference_dense_1_layer_call_and_return_conditional_losses_79900¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÇtrace_0
"
_generic_user_object
"
_generic_user_object
.
A0
B1"
trackable_list_wrapper
 :@@2dense_1/kernel
:@2dense_1/bias
<
M0
N1
O2
P3"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
ß
Ítrace_0
Îtrace_12¤
5__inference_batch_normalization_1_layer_call_fn_79913
5__inference_batch_normalization_1_layer_call_fn_79926³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÍtrace_0zÎtrace_1

Ïtrace_0
Ðtrace_12Ú
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_79946
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_79980³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÏtrace_0zÐtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
Ç
Ötrace_0
×trace_12
)__inference_dropout_1_layer_call_fn_79985
)__inference_dropout_1_layer_call_fn_79990³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÖtrace_0z×trace_1
ý
Øtrace_0
Ùtrace_12Â
D__inference_dropout_1_layer_call_and_return_conditional_losses_79995
D__inference_dropout_1_layer_call_and_return_conditional_losses_80007³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zØtrace_0zÙtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
í
ßtrace_02Î
'__inference_re_lu_1_layer_call_fn_80012¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zßtrace_0

àtrace_02é
B__inference_re_lu_1_layer_call_and_return_conditional_losses_80028¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zàtrace_0
"
_generic_user_object
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
í
ætrace_02Î
'__inference_dense_2_layer_call_fn_80037¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zætrace_0

çtrace_02é
B__inference_dense_2_layer_call_and_return_conditional_losses_80279¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zçtrace_0
"
_generic_user_object
"
_generic_user_object
.
e0
f1"
trackable_list_wrapper
 :@@2dense_2/kernel
:@2dense_2/bias
<
q0
r1
s2
t3"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ènon_trainable_variables
élayers
êmetrics
 ëlayer_regularization_losses
ìlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
ß
ítrace_0
îtrace_12¤
5__inference_batch_normalization_2_layer_call_fn_80292
5__inference_batch_normalization_2_layer_call_fn_80305³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zítrace_0zîtrace_1

ïtrace_0
ðtrace_12Ú
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_80325
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_80359³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zïtrace_0zðtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ñnon_trainable_variables
òlayers
ómetrics
 ôlayer_regularization_losses
õlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
Ç
ötrace_0
÷trace_12
)__inference_dropout_2_layer_call_fn_80364
)__inference_dropout_2_layer_call_fn_80369³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zötrace_0z÷trace_1
ý
øtrace_0
ùtrace_12Â
D__inference_dropout_2_layer_call_and_return_conditional_losses_80374
D__inference_dropout_2_layer_call_and_return_conditional_losses_80386³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zøtrace_0zùtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
í
ÿtrace_02Î
'__inference_re_lu_2_layer_call_fn_80391¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÿtrace_0

trace_02é
B__inference_re_lu_2_layer_call_and_return_conditional_losses_80407¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
"
_generic_user_object
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
í
trace_02Î
'__inference_dense_3_layer_call_fn_80416¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02é
B__inference_dense_3_layer_call_and_return_conditional_losses_80658¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
"
_generic_user_object
"
_generic_user_object
0
0
1"
trackable_list_wrapper
 :@2dense_3/kernel
:2dense_3/bias
J
+0
,1
O2
P3
s4
t5"
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÿBü
-__inference_my_sequential_layer_call_fn_76545input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
-__inference_my_sequential_layer_call_fn_77076inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
-__inference_my_sequential_layer_call_fn_77121inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
-__inference_my_sequential_layer_call_fn_76864input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_my_sequential_layer_call_and_return_conditional_losses_78164inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_my_sequential_layer_call_and_return_conditional_losses_79270inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_my_sequential_layer_call_and_return_conditional_losses_76921input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_my_sequential_layer_call_and_return_conditional_losses_76978input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÊBÇ
#__inference_signature_wrapper_77031input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ÙBÖ
%__inference_dense_layer_call_fn_79279inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ôBñ
@__inference_dense_layer_call_and_return_conditional_losses_79521inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
øBõ
3__inference_batch_normalization_layer_call_fn_79534inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
3__inference_batch_normalization_layer_call_fn_79547inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
N__inference_batch_normalization_layer_call_and_return_conditional_losses_79567inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
N__inference_batch_normalization_layer_call_and_return_conditional_losses_79601inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ìBé
'__inference_dropout_layer_call_fn_79606inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ìBé
'__inference_dropout_layer_call_fn_79611inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
B__inference_dropout_layer_call_and_return_conditional_losses_79616inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
B__inference_dropout_layer_call_and_return_conditional_losses_79628inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ÙBÖ
%__inference_re_lu_layer_call_fn_79633inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ôBñ
@__inference_re_lu_layer_call_and_return_conditional_losses_79649inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ÛBØ
'__inference_dense_1_layer_call_fn_79658inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_dense_1_layer_call_and_return_conditional_losses_79900inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
úB÷
5__inference_batch_normalization_1_layer_call_fn_79913inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
5__inference_batch_normalization_1_layer_call_fn_79926inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_79946inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_79980inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
îBë
)__inference_dropout_1_layer_call_fn_79985inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
îBë
)__inference_dropout_1_layer_call_fn_79990inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
D__inference_dropout_1_layer_call_and_return_conditional_losses_79995inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
D__inference_dropout_1_layer_call_and_return_conditional_losses_80007inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ÛBØ
'__inference_re_lu_1_layer_call_fn_80012inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_re_lu_1_layer_call_and_return_conditional_losses_80028inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ÛBØ
'__inference_dense_2_layer_call_fn_80037inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_dense_2_layer_call_and_return_conditional_losses_80279inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
úB÷
5__inference_batch_normalization_2_layer_call_fn_80292inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
5__inference_batch_normalization_2_layer_call_fn_80305inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_80325inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_80359inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
îBë
)__inference_dropout_2_layer_call_fn_80364inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
îBë
)__inference_dropout_2_layer_call_fn_80369inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
D__inference_dropout_2_layer_call_and_return_conditional_losses_80374inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
D__inference_dropout_2_layer_call_and_return_conditional_losses_80386inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ÛBØ
'__inference_re_lu_2_layer_call_fn_80391inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_re_lu_2_layer_call_and_return_conditional_losses_80407inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ÛBØ
'__inference_dense_3_layer_call_fn_80416inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_dense_3_layer_call_and_return_conditional_losses_80658inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
#:!v@2Adam/dense/kernel/m
:@2Adam/dense/bias/m
,:*@2 Adam/batch_normalization/gamma/m
+:)@2Adam/batch_normalization/beta/m
%:#@@2Adam/dense_1/kernel/m
:@2Adam/dense_1/bias/m
.:,@2"Adam/batch_normalization_1/gamma/m
-:+@2!Adam/batch_normalization_1/beta/m
%:#@@2Adam/dense_2/kernel/m
:@2Adam/dense_2/bias/m
.:,@2"Adam/batch_normalization_2/gamma/m
-:+@2!Adam/batch_normalization_2/beta/m
%:#@2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
#:!v@2Adam/dense/kernel/v
:@2Adam/dense/bias/v
,:*@2 Adam/batch_normalization/gamma/v
+:)@2Adam/batch_normalization/beta/v
%:#@@2Adam/dense_1/kernel/v
:@2Adam/dense_1/bias/v
.:,@2"Adam/batch_normalization_1/gamma/v
-:+@2!Adam/batch_normalization_1/beta/v
%:#@@2Adam/dense_2/kernel/v
:@2Adam/dense_2/bias/v
.:,@2"Adam/batch_normalization_2/gamma/v
-:+@2!Adam/batch_normalization_2/beta/v
%:#@2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v¡
 __inference__wrapped_model_75154} !,)+*DEPMONhitqsr0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿv
ª "1ª.
,
dense_3!
dense_3ÿÿÿÿÿÿÿÿÿ¶
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_79946bPMON3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¶
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_79980bOPMN3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_batch_normalization_1_layer_call_fn_79913UPMON3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@
5__inference_batch_normalization_1_layer_call_fn_79926UOPMN3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@¶
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_80325btqsr3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¶
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_80359bstqr3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
5__inference_batch_normalization_2_layer_call_fn_80292Utqsr3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@
5__inference_batch_normalization_2_layer_call_fn_80305Ustqr3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@´
N__inference_batch_normalization_layer_call_and_return_conditional_losses_79567b,)+*3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ´
N__inference_batch_normalization_layer_call_and_return_conditional_losses_79601b+,)*3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
3__inference_batch_normalization_layer_call_fn_79534U,)+*3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@
3__inference_batch_normalization_layer_call_fn_79547U+,)*3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@¢
B__inference_dense_1_layer_call_and_return_conditional_losses_79900\DE/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 z
'__inference_dense_1_layer_call_fn_79658ODE/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@¢
B__inference_dense_2_layer_call_and_return_conditional_losses_80279\hi/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 z
'__inference_dense_2_layer_call_fn_80037Ohi/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@¤
B__inference_dense_3_layer_call_and_return_conditional_losses_80658^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_3_layer_call_fn_80416Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ 
@__inference_dense_layer_call_and_return_conditional_losses_79521\ !/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿv
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 x
%__inference_dense_layer_call_fn_79279O !/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿv
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dropout_1_layer_call_and_return_conditional_losses_79995\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¤
D__inference_dropout_1_layer_call_and_return_conditional_losses_80007\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
)__inference_dropout_1_layer_call_fn_79985O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@|
)__inference_dropout_1_layer_call_fn_79990O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dropout_2_layer_call_and_return_conditional_losses_80374\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¤
D__inference_dropout_2_layer_call_and_return_conditional_losses_80386\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
)__inference_dropout_2_layer_call_fn_80364O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@|
)__inference_dropout_2_layer_call_fn_80369O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@¢
B__inference_dropout_layer_call_and_return_conditional_losses_79616\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¢
B__inference_dropout_layer_call_and_return_conditional_losses_79628\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 z
'__inference_dropout_layer_call_fn_79606O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@z
'__inference_dropout_layer_call_fn_79611O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@Å
H__inference_my_sequential_layer_call_and_return_conditional_losses_76921y !,)+*DEPMONhitqsr8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿv
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
H__inference_my_sequential_layer_call_and_return_conditional_losses_76978y !+,)*DEOPMNhistqr8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿv
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
H__inference_my_sequential_layer_call_and_return_conditional_losses_78164x !,)+*DEPMONhitqsr7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿv
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
H__inference_my_sequential_layer_call_and_return_conditional_losses_79270x !+,)*DEOPMNhistqr7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿv
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_my_sequential_layer_call_fn_76545l !,)+*DEPMONhitqsr8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿv
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_my_sequential_layer_call_fn_76864l !+,)*DEOPMNhistqr8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿv
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_my_sequential_layer_call_fn_77076k !,)+*DEPMONhitqsr7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿv
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_my_sequential_layer_call_fn_77121k !+,)*DEOPMNhistqr7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿv
p

 
ª "ÿÿÿÿÿÿÿÿÿ
B__inference_re_lu_1_layer_call_and_return_conditional_losses_80028X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 v
'__inference_re_lu_1_layer_call_fn_80012K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@
B__inference_re_lu_2_layer_call_and_return_conditional_losses_80407X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 v
'__inference_re_lu_2_layer_call_fn_80391K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@
@__inference_re_lu_layer_call_and_return_conditional_losses_79649X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 t
%__inference_re_lu_layer_call_fn_79633K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@°
#__inference_signature_wrapper_77031 !,)+*DEPMONhitqsr;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿv"1ª.
,
dense_3!
dense_3ÿÿÿÿÿÿÿÿÿ