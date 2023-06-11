import styled from "styled-components";
const Spacer = styled.div`
  width: ${(props) => (props.width ? props.width : "auto")};
  height: ${(props) => (props.height ? props.height : "auto")};
`;
export { Spacer };
