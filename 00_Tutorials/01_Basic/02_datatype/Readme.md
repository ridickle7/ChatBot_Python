# 1-2. Datatype

텐서플로우의 자료형은 뉴럴네트워크에 최적화되어 있는 개발 Framework  
-> 그 자료형과, 실행 방식이 약간 일반적인 프로그래밍 방식과 상이하다.


### 자료형 종류
#### 1. Constant (상수형)  
말 그대로 상수를 이야기하며, 정의 및 사용 예제는 다음과 같다.
<pre><code># value : 상수의 값
# dtype : 상수의 데이타형 (ex> tf.float32)
# shape : 행렬의 차원(ex> shape=[3,3] -> 3x3 행렬을 저장)
# name  : 상수의 이름. 

tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)</code></pre>
재미를 붙여 사칙연산을 해 보았으나 우리는 이상한 결과를 확인할 수 있다.
<pre><code># 예상치 못한 값!

import tensorflow as tf

a = tf.constant([5], dtype=tf.float32)
b = tf.constant([10], dtype=tf.float32)
c = tf.constant([2], dtype=tf.float32)

d = a * b + c

# '52' 가 아닌 'Tensor("add:0", shape=(1,), dtype=float32)' 가 나온다
print(d)
</code></pre>  
도데체 이 녀석의 정체는 무엇일까? 답은 아래와 같다.
#### 2. Graph & Session  
위의 d 값은 계산 값이 아닌 a * b + c 연산 식 (그래프) 을 정의 하는 것이다.
![Image](http://cfile8.uf.tistory.com/image/221D7F45584AB42A1F0F4F)  
그럼 본론으로 돌아와 계산 값을 뽑아내려면 연산식에 a, b, c 값을 각각 넣어야하는데 이는 세션을 생성하여 그래프를 실행해야 한다.
코드로 설명하면 다음과 같다.
<pre><code> # 연산 값 뽑아내기

import tensorflow as tf

a = tf.constant([5],dtype=tf.float32)
b = tf.constant([10],dtype=tf.float32)
c = tf.constant([2],dtype=tf.float32)

d = a*b+c

# 세션을 생성하여 run 내장함수를 실행해야 연산이 진행된다.
# (파라미터로 그래프 value를 넣어 준다.)
sess = tf.Session()     # 세션 생성
result = sess.run(d)    # run 을 통해 해당 연산(파라미터) 실행
print(result)           # 값 확인
</code></pre>
#### 3. PlaceHolder  
입력 값으로 여러 개의 데이터를 그래프에 넣을 경우 해당 학습용 데이터를 담는 그릇을 **placeHolder** 이라 한다.  
정의 및 사용 예제는 다음과 같다.
<pre><code># dtype : 플레이스홀더에 저장되는 데이타형 (ex> tf.float32)
# shape : 행렬의 차원(ex> shape=[3,3] -> 3x3 행렬을 저장)
# name  : 플레이스 홀더의 이름

import tensorflow as tf

input_data = [1,2,3,4,5]
x = tf.placeholder(dtype=tf.float32)
y = x * 2

sess = tf.Session()
result = sess.run(y, **feed_dict={x:input_data}**) # x에 학습용 데이타를 넣어주는 과정 (피딩(feeding))

print(result)
</code></pre>
#### 4. Variable
학습용 가설을 만들었을 때 (ex> y = W * x + b)  
x가 입력데이터였다면 W와 b는 학습을 통해 구해야 하는 값이 된다. (이를 **변수** 라 한다.)  
변수형은 Variable 형의 객체로 생성된다.  
정의 및 사용 예제는 다음과 같다.
<pre><code> # example > y = W * x

import tensorflow as tf

input_data = [1,2,3,4,5]
x = tf.placeholder(dtype=tf.float32)
W = tf.Variable([2],dtype=tf.float32)
y = W*x

sess = tf.Session()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>      # variable 초기화 및 실행
init = tf.global_variables_initializer()
sess.run(init)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

result = sess.run(y,feed_dict={x:input_data})	# error ( Variable 초기화 선 진행 필요! )

print(result)
</code></pre>

### 결론
1. 모델을 그래프로 정의하고
2. 세션을 만들어서 그래프를 실행하고
3. 세션이 실행될때 그래프에 동적으로 값을 넣어가면서 (피딩) 
4. 실행한다 

이 기본 flow 를 잘 이해해야, 텐서플로우 프로그래밍을 제대로 시작할 수 있다.
                  
### 참고 자료
- 조대협의 블로그 : http://bcho.tistory.com/1150

