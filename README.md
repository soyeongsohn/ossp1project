# ossp1project

Rethinking Style Transfer: From Pixels to Parameterized Brushstrokes 논문 구현 프로젝트 (in PyTorch)<br/>
공식 repo <a href="https://github.com/CompVis/brushstroke-parameterized-style-transfer">링크</a> <br/>
<a href="https://github.com/justanhduc/brushstroke-parameterized-style-transfer/blob/master/losses.py">파이토치 링크</a> </br>
* 파이토치 repo 코드를 분석하여 공부하는 방향으로 진행한 팀원이 있어서 해당 링크 같이 올립니다.

<br/>
### Use Case Diagram
![image](https://user-images.githubusercontent.com/60024018/210245234-d2c61960-8208-492e-a022-0d48ea6c39f4.png)

### Software Architecture
![image](https://user-images.githubusercontent.com/60024018/178115865-491b421c-8f10-45b9-b8de-efc1cb30422e.png)

### Result
![image](https://user-images.githubusercontent.com/60024018/178115758-825d991b-1e41-497d-bc78-3aff76e46e2e.png)

좌측 상단이 렌더링이 제대로 진행되지 않음. 하지만 brushstroke의 질감은 살아 있는 느낌. <br/>
<br/>
맡은 부분: brushstroke, renderer, app, code refactoring <br/>

#### streamlit
![image](https://user-images.githubusercontent.com/60024018/178115947-4ef7aabc-2464-4f3d-96a6-49cccf81dced.png)

### 배운점
1. 논문에 나와 있는 수식과 알고리즘을 이해하고 적용해볼 수 있었다.
2. tensorflow로 구현된 코드를 pytorch로 변환하며 pytorch 내장함수를 더 능숙하게 다룰 수 있게 되었다.
3. style transfer의 작동 원리를 배울 수 있었다.

### 아쉬운점
1. 코드 구현 시 막히는 부분이 있었다. 논문에 방법론이 구체적으로 나와있지 않았고, 프로젝트 마감 기한때문에 해당 부분은 tensorflow로 구현된 공식 repo를 참고하여 구현을 진행했다는 점에서 아쉬웠다.
