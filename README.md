## Báo cáo LaTeX chủ đề VCO

### Cấu trúc
- `report.tex`: Tệp LaTeX chính (XeLaTeX) viết tiếng Việt.
- `Makefile`: Lệnh build nhanh.
- `.github/workflows/latex.yml`: Tự động build PDF khi commit (GitHub Actions).
  - Đã cấu hình `biber` cho bibliography.

### Yêu cầu cục bộ
- Cài TeX Live (đầy đủ) hoặc MiKTeX, `latexmk` và `biber`.
- Khuyến nghị dùng XeLaTeX (đã cấu hình sẵn qua `.latexmkrc`).

### Lệnh build
```bash
make pdf     # build report.pdf
make watch   # build tự động khi file thay đổi
make clean   # dọn file trung gian
```

Trên Windows (PowerShell) nếu không có `make`:
```powershell
./build.ps1   # build report.pdf bằng latexmk
```

### CI
- Khi push lên GitHub, workflow sẽ tạo `report.pdf` và đính kèm artifact tên `report-pdf`.
  - Workflow đã cài gói cần thiết (`biblatex`, `biber`, `csquotes`, `siunitx`, `circuitikz`).

### Gợi ý
- Sửa thông tin tên và MSSV trong phần `\author{...}` của `report.tex`.
- Thêm hình ảnh bằng `\includegraphics` (đặt file ảnh cạnh `report.tex`).
  - Ảnh nên để trong thư mục `figures/`.
  - Tham khảo trích dẫn qua `references.bib` và dùng `\cite{...}`.


