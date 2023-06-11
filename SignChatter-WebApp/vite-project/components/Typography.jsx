import styled from "styled-components";
const Wrapper = styled.div``;
const SpanWrapper = styled.span``;
const H1 = styled.h1``;
const H2 = styled.h2``;
const H3 = styled.h3``;
const P = styled.p`
  font-style: normal;
  text-transform: ${(props) => (props.tt ? props.tt : "normal")};
  font-weight: ${(props) => (props.weight ? props.weight : "")};
  line-height: ${(props) => (props.lHeight ? props.lHeight : "")};
  font-size: ${(props) => (props.size ? props.size : "16px")};
  color: ${(props) => (props.color ? props.color : "#ffffff")};
`;
export { SpanWrapper, Wrapper, H1, H2, H3, P };
