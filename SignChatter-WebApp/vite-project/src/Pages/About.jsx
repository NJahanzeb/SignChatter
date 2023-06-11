import React from "react";
import { Container } from "../../components/Layout";
import { Wrapper,P } from "../../components/Typography";
const About = (props) => {
  return (
    <>
      <Container>
        <Container className="d-flex flex-row justify-content-center align-items-center" style={{width:"80%"}}
        >
          <Wrapper className="mt-5">
          <P size="16px" color="black" weight="600" lHeight="21px">SignChatter is the world’s first sign language translation app that takes body movement into account while detecting and translating signs hence, giving more accurate translations and being able to translate a lot more signs than singular-frame based sign language translation apps. The app currently supports only American Sign Language but support for other languages will be added soon.</P>
          </Wrapper>
         
        </Container>
      </Container>
    </>
  );
};

export default About;
