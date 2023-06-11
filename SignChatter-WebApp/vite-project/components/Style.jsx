import styled from "styled-components";
import Button from "@mui/material/Button";
const videoConstraints = {
  width: 720,
  height: 400,
  facingMode: "user",
};
const BackgroundCamera = styled.div`
  position: relative;
  background: gray;
  width: 720px;
  height: 400px;
`;
const BackgroundTranslation = styled.div`
  background: black;
  border: 1px solid white;
  border-radius: 15px;
  color: white;
  padding-top: 15px;
  padding-bottom: 15px;
  padding-left: 20px;
  padding-right: 20px;
  width: 720px;
  height: 150px;
`;
const ButtonStyle = styled(Button)`
  border-radius: 50% !important;
  height: 65px;
  width: 30px;
`;
const SearchButton = styled.div`
  background: #fd7e14;
  height: 55px;
  border: 1px solid black;
`;
const RedDot = styled.div`
  background: red;
  border-radius: 37%;
  height: 16px;
  width: 16px;
`;
const RedDotOpen = styled.div`
  background: red;
  border-radius: 50%;
  height: 12px;
  width: 12px;
`;
export {
  videoConstraints,
  BackgroundCamera,
  BackgroundTranslation,
  ButtonStyle,
  SearchButton,
  RedDot,
  RedDotOpen
};
