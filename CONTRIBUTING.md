Terima kasih telah tertarik berkontribusi pada **Teman Isyarat**! Proyek ini mengembangkan teknologi pengenalan bahasa isyarat BISINDO (Bahasa Isyarat Indonesia) yang inklusif dan dapat diakses semua orang. 
## Daftar Isi
- [Jenis Kontribusi](#jenis-kontribusi)
- [Proses Berkontribusi](#proses-berkontribusi)
- [Git Workflow](#git-workflow)
- [Pull Request](#pull-request)
## Jenis Kontribusi

| Jenis           | Deskripsi                              | Contoh                                     |     |
| --------------- | -------------------------------------- | ------------------------------------------ | --- |
| **Bug Reports** | Laporkan bug dengan detail reproduksi  | Crash preprocessing, model inference error |     |
| **Features**    | Saran fitur baru dengan use case jelas | Real-time detection, mobile app support    |     |
| **Code**        | Refactor, optimasi, test coverage      | Performance improvements, bug fixes        |     |
| **Docs**        | README, tutorial, API docs             | Setup guides, model documentation          |     |
| **Dataset/ML**  | BISINDO dataset, models, benchmarks    | New gesture classes, improved accuracy     |     |
## Proses Berkontribusi
### 1. Fork & Clone
```bash
# Fork di GitHub: https://github.com/williamu04/temanisyarat
git clone https://github.com/YOUR_USERNAME/temanisyarat.git
cd temanisyarat
git remote add upstream https://github.com/williamu04/temanisyarat.git
```
### 2. Buat Issue Dulu (WAJIB untuk fitur besar)
- [Issues](https://github.com/williamu04/temanisyarat/issues)
- Hindari duplikasi kerja
### 3. Branching (JANGAN push ke main)
```bash
git checkout main && git pull upstream main
git checkout -b TYPE/nama-deskriptif
```
**Naming convention:**
```
feature/gesture-detection
fix/preprocessing-crash  
docs/installation-guide
refactor/model-loader
test/unit-tests
```
### 4. Update Personal Log
Update `logs/{nama_anda}.md`:
```markdown
17 Maret 2026
- Implementasi model baru
- Update docs
- Menggunakan MediaPipe Hands
```
**PENTING**: Log pribadi masuk dalam PR feature branch Anda 
### 5. Commit Standards
```bash
git commit -m "feat(detection): tambah model MediaPipe Hands
- improve accuracy 15%
- real-time processing <30ms
Closes #12"
```
**Types**: `feat` | `fix` | `docs` | `style` | `refactor` | `perf` | `test`
### 6. Push \& PR
```bash
git push origin feature/nama-fitur
# → GitHub akan prompt buat PR
```
## Git Workflow
```
Fork repo → Clone → Issue → Feature branch → Code+Log → Push → PR → Review → Merge
         ↑
      git pull upstream main (keep sync)
```
**Sync conflicts:**
```bash
git fetch upstream
git rebase upstream/main
git push --force-with-lease origin feature/nama-fitur
```
## Pull Request
### Title Format
```
[ML] Model MediaPipe Hands +15% accuracy
[BUG] Fix preprocessing memory leak  
[DOCS] Installation guide lengkap
```
### Template PR Description

```markdown
Description
[Apa yang diubah dan mengapa]

Changes
- [ ] Feature/bugfix X
- [ ] Tests pass
- [ ] Docs updated
- [ ] logs/{nama}.md 

Related: Closes #12

Testing:
- [ ] Unit tests 
- [ ] Manual test
- [ ] No new warnings
```
## Success Tips

| Do ✓               | Don't ✗                  |
| ------------------ | ------------------------ |
| Start small issues | Push langsung ke main    |
| Read existing code | Skip issue discussion    |
| Update `logs/*.md` | Ignore reviewer feedback |
| Write tests        | Vague commit messages    |
| Ask if unclear     | Merge tanpa approval     |
