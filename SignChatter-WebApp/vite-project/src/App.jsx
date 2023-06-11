import React, { useState } from "react";
import { Routes, Route,useNavigate } from "react-router-dom";
import Home from "./Pages/Home";
import ASL from "./Pages/ASL";
import About from "./Pages/About";
import Instructions from "./Pages/Instructions";
import Error from "./Pages/404";
import DrawerComp from "../components/Drawer";
import { Wrapper } from "../components/Typography";
import { Container, Image } from "../components/Layout";
import Logo from "./assets/logo.png";
function App() {
  const router=useNavigate();
  const [page, setPage] = useState(0);
  return (
    <>
      <Container>
        <Wrapper
          className="d-flex flex-row align-items-center justify-content-between mt-5"
          style={{ marginRight: "1.7%" }}
        >
          <DrawerComp page={page} setPage={setPage} />
          <Image
          onClick={()=>{router("/")}}
          style={{cursor:"pointer"}}
            className="img-fluid"
            src={Logo}
            alt="Logo"
            width="100px"
            height="100px"
          />
          <div className="me-5" />
        </Wrapper>
      </Container>
      <Routes>
        <Route exact path="/" element={<Home page={page} setPage={setPage}/>} />
        <Route exact path="/asld" element={<ASL page={page} setPage={setPage}/>} />
        <Route exact path="/about" element={<About page={page} setPage={setPage}/>} />
        <Route exact path="/instructions" element={<Instructions page={page} setPage={setPage}/>} />
        <Route path="/*" element={<Error/>} />
      </Routes>
    </>
  );
}

export default App;
