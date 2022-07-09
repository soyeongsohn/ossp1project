# ossp1project

Rethinking Style Transfer: From Pixels to Parameterized Brushstrokes 논문 구현 프로젝트 (in PyTorch)<br/>
공식 repo <a href="https://github.com/CompVis/brushstroke-parameterized-style-transfer">링크</a> <br/>
<br/>
### Use Case Diagram
![image](https://user-images.githubusercontent.com/60024018/178115869-d0ea7317-8a52-41d8-bb55-6bc63c850a19.png)

### Software Architecture
![image](https://user-images.githubusercontent.com/60024018/178115865-491b421c-8f10-45b9-b8de-efc1cb30422e.png)

### Result
![image](https://user-images.githubusercontent.com/60024018/178115758-825d991b-1e41-497d-bc78-3aff76e46e2e.png)

좌측 상단이 렌더링이 제대로 진행되지 않음. 하지만 brushstroke의 질감은 살아 있는 느낌. <br/>
<br/>
맡은 부분: brushstroke, renderer, app <br/>

#### streamlit
![image](https://user-images.githubusercontent.com/60024018/178115947-4ef7aabc-2464-4f3d-96a6-49cccf81dced.png)
