import React from 'react';
import { Container } from "../../components/Layout";
import { Wrapper,P } from "../../components/Typography";

const instructions=[
    "Stand about 2-2.5m (6-8ft) away from the camera",
"Perform only one sign per video and make sure the entire sign is recorded in the video",
"Make sure there is plenty of light while making the video",
"Ensure there isn’t any other person present in the background",
"Videos should be about 3-6 seconds in length",
"Please stand while recording the sign and make sure your body is visible till at least knee level"

]
const Instructions = (props) => {
  return (
    <>
          <Container>
        <Container className="d-flex flex-row justify-content-start align-items-center mt-5" style={{width:"80%"}}
        >
          <Wrapper className="mt-5">
          <ol>
          {
            instructions.map((val,index)=>{
                return( 
                    <>
                    <li>
                    <P size="16px" color="black" weight="600" lHeight="21px" key={index}>{val}</P>
                    </li>
                    </>
                )
            })
          }
          </ol>
          </Wrapper>
         
        </Container>
      </Container>
    </>
  )
}
export default Instructions