import React, { useState } from "react";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import { styled } from "@mui/material/styles";
export default function ToggleBtn(props) {
  const [alignment, setAlignment] = useState("asl");

  const handleChange = (event, newAlignment) => {
    setAlignment(newAlignment);
  };
  return (
    <div style={{ borderRadius: "50%" }}>
      <ToggleButtonGroup
        color="standard"
        sx={{
          background: "#F58F20",
          color: "white",
          weight: "900",
        }}
        value={alignment}
        exclusive
        onChange={handleChange}
        aria-label="Platform"
      >
        <ToggleButton
          value="asl"
          sx={{ color: "white", background: "black", weight: "900" }}
          onClick={() => {
            props.setPage(1);
          }}
        >
          ASL
        </ToggleButton>
        <ToggleButton
          value="psl"
          sx={{ color: "white", background: "black", weight: "900" }}
          onClick={() => {
            props.setPage(0);
          }}
        >
          PSL
        </ToggleButton>
      </ToggleButtonGroup>
    </div>
  );
}
