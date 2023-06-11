import * as React from "react";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import Button from "@mui/material/Button";
import List from "@mui/material/List";
import Divider from "@mui/material/Divider";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import InboxIcon from "@mui/icons-material/MoveToInbox";
import MailIcon from "@mui/icons-material/Mail";
import MenuIcon from "@mui/icons-material/Menu";
import InfoIcon from "@mui/icons-material/Info";
import LabelIcon from "@mui/icons-material/Label";
import { useNavigate } from "react-router-dom";
export default function DrawerComp(props) {
  const router=useNavigate();
  const [state, setState] = React.useState({
    top: false,
    left: false,
    bottom: false,
    right: false,
  });

  const toggleDrawer = (anchor, open) => (event) => {
    if (
      event.type === "keydown" &&
      (event.key === "Tab" || event.key === "Shift")
    ) {
      return;
    }

    setState({ ...state, [anchor]: open });
  };

  const list = (anchor) => (
    <Box
      sx={{
        width: anchor === "top" || anchor === "bottom" ? "auto" : 250,
        height: "100%",
        marginTop: "15%",
      }}
      role="presentation"
      onClick={toggleDrawer(anchor, false)}
      onKeyDown={toggleDrawer(anchor, false)}
    >
      {/* <h1 className="ms-3 mt-5 mb-5">Menu</h1> */}

      <List>
        {[ "ASL Dict","Instructions"].map((text, index) => (
          <ListItem
            key={text}
            disablePadding
            className="mt-2"
            onClick={() => {
              // props.setPage(index);
              router(index==0?"/asld":"instructions");
            }}
          >
            <ListItemButton>
              <ListItemIcon>
                <LabelIcon style={{ color: "#FD7E14" }} className="mt-1" />
              </ListItemIcon>
              <p
                className="mb-0"
                style={{ color: "black", fontWeight: "700", fontSize: "17px" }}
              >
                {text}
              </p>
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      <Divider sx={{ background: "black" }} />
      <List>
        {["About"].map((text, index) => (
          <ListItem
            key={text}
            disablePadding
            className="mt-1"
            onClick={() => {
              router("/about");
              // props.setPage(1);
            }}
          >
            <ListItemButton>
              <ListItemIcon style={{ marginTop: "5px" }}>
                <InfoIcon style={{ color: "#FD7E14" }} />
              </ListItemIcon>
              <p
                className="mb-0"
                style={{ color: "black", fontWeight: "700", fontSize: "17px" }}
              >
                {text}
              </p>
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Box>
  );

  return (
    <div>
      {["left"].map((anchor) => (
        <React.Fragment key={anchor}>
          <Button onClick={toggleDrawer(anchor, true)}>
            <MenuIcon style={{ color: "#000000", weight: 900 }} />
          </Button>
          <Drawer
            anchor={anchor}
            open={state[anchor]}
            onClose={toggleDrawer(anchor, false)}
          >
            {list(anchor)}
          </Drawer>
        </React.Fragment>
      ))}
    </div>
  );
}
