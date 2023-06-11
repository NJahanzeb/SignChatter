import { Container, Row, Col } from "react-bootstrap";
import styled from "styled-components";
import Image from "react-bootstrap/Image";
const BoxComp = styled.div`
  background: ${(props) => (props.bg ? props.bg : "")};
  border: ${(props) => (props.border ? props.border : "")};
  border-radius: ${(props) => (props.borderRadius ? props.borderRadius : "")};
  height: ${(props) => (props.height ? props.height : "")};
  width: ${(props) => (props.width ? props.width : "auto")};
  backdrop-filter: ${(props) => (props.bFilter ? props.bFilter : "")};
`;
export { Container, Row, Col, Image, BoxComp };
