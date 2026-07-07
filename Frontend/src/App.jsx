import { useState, useEffect } from "react";
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Card,
  CardContent,
  Paper,
  CardActionArea,
  CardMedia,
  Grid,
  TableContainer,
  Table,
  TableBody,
  TableHead,
  TableRow,
  TableCell,
  Button,
  CircularProgress,
  Box,
} from "@mui/material";
import { Clear } from "@mui/icons-material";
import { styled } from "@mui/material/styles";
import axios from "axios";
import bgImage from "./assets/bg.png";

// ============================================
// STYLED COMPONENTS
// ============================================

const StyledAppBar = styled(AppBar)({
  background: "#be6a77",
  boxShadow: "none",
  color: "white",
});

const MainContainer = styled(Container)({
  backgroundImage: `url(${bgImage})`,
  backgroundRepeat: "no-repeat",
  backgroundPosition: "center",
  backgroundSize: "cover",
  height: "93vh",
  marginTop: "8px",
  display: "flex",
  justifyContent: "center",
  alignItems: "flex-start",
  paddingTop: "4em",
});

const ImageCard = styled(Card, {
  shouldForwardProp: (prop) => prop !== "hasimage",
})(({ hasimage }) => ({
  margin: "auto",
  maxWidth: 400,
  height: hasimage === "true" ? 500 : "auto",
  backgroundColor: "transparent",
  boxShadow: "0px 9px 70px 0px rgb(0 0 0 / 30%) !important",
  borderRadius: "15px",
}));

const DetailBox = styled(Box)({
  backgroundColor: "white",
  display: "flex",
  justifyContent: "center",
  flexDirection: "column",
  alignItems: "center",
  padding: "16px",
});

const ClearButton = styled(Button)({
  width: "100%",
  borderRadius: "15px",
  padding: "15px 22px",
  color: "#000000a6",
  fontSize: "20px",
  fontWeight: 900,
  backgroundColor: "white",
  "&:hover": {
    backgroundColor: "#ffffff7a",
  },
});

const TransparentTable = styled(TableContainer)({
  backgroundColor: "transparent !important",
  boxShadow: "none !important",
});

const TransparentTableCell = styled(TableCell)({
  fontSize: "22px",
  backgroundColor: "transparent !important",
  borderColor: "transparent !important",
  color: "#000000a6 !important",
  fontWeight: "bolder",
  padding: "1px 24px 1px 16px",
});

const SmallTableCell = styled(TableCell)({
  fontSize: "14px",
  backgroundColor: "transparent !important",
  borderColor: "transparent !important",
  color: "#000000a6 !important",
  fontWeight: "bolder",
  padding: "1px 24px 1px 16px",
});

const UploadBox = styled(Box)({
  border: "2px dashed #ccc",
  borderRadius: "15px",
  padding: "40px",
  textAlign: "center",
  cursor: "pointer",
  backgroundColor: "rgba(255,255,255,0.9)",
  "&:hover": {
    borderColor: "#be6a77",
    backgroundColor: "rgba(255,255,255,1)",
  },
});

// ============================================
// COMPONENT UTAMA
// ============================================

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [data, setData] = useState(null);
  const [image, setImage] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const sendFile = async () => {
    if (!image || !selectedFile) return;

    let formData = new FormData();
    formData.append("file", selectedFile);

    try {
      let res = await axios({
        method: "post",
        url: "http://localhost:8000/predict",
        data: formData,
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      if (res.status === 200) {
        setData(res.data);
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Gagal prediksi! Pastikan API berjalan di port 8000");
    } finally {
      setIsLoading(false);
    }
  };

  const clearData = () => {
    setData(null);
    setImage(false);
    setSelectedFile(null);
    setPreview(null);
  };

  useEffect(() => {
    if (!selectedFile) {
      setPreview(null);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
  }, [selectedFile]);

  useEffect(() => {
    if (!preview) return;
    setIsLoading(true);
    sendFile();
  }, [preview]);

  const handleFileChange = (event) => {
    const files = event.target.files;
    if (!files || files.length === 0) {
      setSelectedFile(null);
      setImage(false);
      setData(null);
      return;
    }
    setSelectedFile(files[0]);
    setData(null);
    setImage(true);
  };

  // ✅ PERUBAHAN: confidence sudah persen dari API, tidak perlu × 100
  let confidence = 0;
  if (data) {
    confidence = parseFloat(data.confidence).toFixed(2);
  }

  return (
    <>
      <StyledAppBar position="static">
        <Toolbar>
          <Typography variant="h6" noWrap sx={{ flexGrow: 1 }}>
            Potato Disease Classification
          </Typography>
        </Toolbar>
      </StyledAppBar>

      <MainContainer maxWidth={false} disableGutters>
        <Grid
          container
          direction="row"
          justifyContent="center"
          alignItems="center"
          spacing={2}
        >
          <Grid item xs={12}>
            <ImageCard hasimage={image.toString()}>
              {image && preview && (
                <CardActionArea>
                  <CardMedia
                    component="img"
                    image={preview}
                    alt="Preview"
                    sx={{ height: 400, objectFit: "cover" }}
                  />
                </CardActionArea>
              )}

              {!image && (
                <CardContent>
                  <UploadBox>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      style={{ display: "none" }}
                      id="file-input"
                    />
                    <label htmlFor="file-input" style={{ cursor: "pointer" }}>
                      <Typography variant="h6">
                        📁 Klik untuk Upload
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        atau drag & drop gambar daun kentang
                      </Typography>
                    </label>
                  </UploadBox>
                </CardContent>
              )}

              {data && (
                <DetailBox>
                  <TransparentTable component={Paper}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <SmallTableCell>Label:</SmallTableCell>
                          <SmallTableCell align="right">Confidence:</SmallTableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        <TableRow>
                          <TransparentTableCell component="th" scope="row">
                            {data.class}
                          </TransparentTableCell>
                          <TransparentTableCell align="right">
                            {confidence}%
                          </TransparentTableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TransparentTable>
                </DetailBox>
              )}

              {isLoading && (
                <DetailBox>
                  <CircularProgress sx={{ color: "#be6a77 !important" }} />
                  <Typography variant="h6" sx={{ mt: 2 }}>
                    Processing...
                  </Typography>
                </DetailBox>
              )}
            </ImageCard>
          </Grid>

          {data && (
            <Grid item sx={{ maxWidth: "416px", width: "100%" }}>
              <ClearButton
                variant="contained"
                size="large"
                onClick={clearData}
                startIcon={<Clear fontSize="large" />}
              >
                Clear
              </ClearButton>
            </Grid>
          )}
        </Grid>
      </MainContainer>
    </>
  );
}

export default App;