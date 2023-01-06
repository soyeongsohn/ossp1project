## 기존 문제

![image](https://user-images.githubusercontent.com/60024018/210918370-eaa75e0c-0b69-4d70-8cfd-08a6af4615b6.png) <br/>
좌상단 렌더링이 제대로 되지 않음 <br/>

## 개선 방법
default 값으로 돌려가며 육안으로 비교하기 <br/>

## 개선 일지

### 12.28.2022
기존 코드에서 content loss 계산 시 사용되는 VGG의 convolution layer 번호가 잘못 되어있다는 것을 발견 <br/>
수정하고 다시 돌려봤으나 문제 해결 못함

### 01.05.2023
범위 안에 있어야 할 것들의 값 찍어보기 -> 범위 벗어나면 clipping하기 <br/>
![image](https://user-images.githubusercontent.com/60024018/210918953-380019d7-ef03-4374-8a8c-58e648e3914d.png) <br/>
튀는 부분들이 조금 개선되었다. <br/>

### 01.06.2023
full canvas에 대한 k개의 근접 brushstroke line segment를 구할 때 start point를 빼서 구해야하는데 start point와 end point의 거리를 뺐었다. <br/>
이를 바꿔주며 기존 shape 오류로 인해 추가했던 코드들을 삭제하였다. </br>
![image](https://user-images.githubusercontent.com/60024018/210921040-26486b5d-99de-4c70-8a17-6bc2b3b399c1.png) <br/>
인덱스가 0에 가까운 부분들에 문제가 있는 것 같다. color를 찍어보니 \[0, 0, 1](파란색)로 수렴하여서 코드를 확인해봤는데, brushstroke curve의 segment 위 좌표점에서의 projection을 구하는 과정에서 projection이 0과 1 사이 밖인 것을 clipping을 해주지 않아서 생긴 문제인 것 같다.<br/>
clipping을 해준 결과 처음과 비교했을 때 많이 개선된 결과를 얻을 수 있었다. <br/>
![image](https://user-images.githubusercontent.com/60024018/210931370-b37106a8-9763-4bc0-89bb-00fc454cc50b.png) <br/>

<br/>
추가로 bezier curve 함수를 논문 공식대로 구현한 것에서 공식 repo에서 구현한 것처럼 바꿔주었고, brushstroke를 initialize하는 과정에서 중심점을 구하는 것을 중앙값에서 평균값으로 바꿔주었다. </br>

![image](https://user-images.githubusercontent.com/60024018/210932870-27fe12c6-a429-44c5-9274-5f0a756d6149.png)
