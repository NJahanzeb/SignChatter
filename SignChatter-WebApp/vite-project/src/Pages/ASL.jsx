import React from "react";
import { Wrapper,P } from "../../components/Typography";
import TextField from "@mui/material/TextField";
import { Container} from "../../components/Layout";
import SearchIcon from "@mui/icons-material/Search";
import { Spacer } from "../../components/Spacer";
import { SearchButton } from "../../components/Style";

const words=[
  "Banana",
"Bar",
"Basement",
"Basketball",
"Bath",
"Bathroom",
"Bear",
"Beard",
"Bed",
"Bedroom"
]
const ASL = (props) => {
  return (
    <>
    <Container>
    <Spacer height="30px"/>
      <Wrapper className="d-flex flex-row justify-content-center">
        <TextField
          id="outlined-basic"
          label="ASL Dictionary"
          variant="outlined"
          value={""}
          sx={{
            width: "400px",
            color: "black",
          }}
        />
        <SearchButton>
          <SearchIcon
            className="ms-2 me-1 mt-1"
          variant="standard"
            style={{
              width: "50px",
              height: "50px",
              color: "black",
              cursor: "pointer",
              outline:"none"
            }}
          />
        </SearchButton>
      </Wrapper>
      </Container>
      <Container>
      <Container className="d-flex flex-row align-items-center justify-content-start mt-5" style={{width:"70%"}}
      >
        <Wrapper className="mt-1"> 
        <ul >
          {
            words.map((val,index)=>{
                return( 
                    <>
                    <li>
                    <P size="16px" color="black" weight="600" lHeight="21px" key={index}>{val}</P>
                    </li>
                    </>
                )
            })
          }
          </ul>
          </Wrapper>
      </Container>
      </Container>
    </>
  );
};

export default ASL;
