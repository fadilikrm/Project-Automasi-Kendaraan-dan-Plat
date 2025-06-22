// Wait for the DOM to be fully loaded before running the script
document.addEventListener("DOMContentLoaded", () => {
  // --- DOM Elements ---
  // Login Page
  const loginPage = document.getElementById("login-page");
  const loginForm = document.getElementById("login-form");
  const loginError = document.getElementById("login-error");
  const usernameInput = document.getElementById("username");
  const passwordInput = document.getElementById("password");

  // App Container
  const appContainer = document.getElementById("app-container");
  const pageTitle = document.getElementById("page-title");
  const userNameDisplay = document.getElementById("user-name");
  const userAvatarDisplay = document.getElementById("user-avatar");
  const logoutBtn = document.getElementById("logout-btn");

  // Navigation Links & Sections
  const navLinks = document.querySelectorAll(".nav-link");
  const pageSections = document.querySelectorAll(".page-section");
  const navPengaturanSistem = document.getElementById("nav-pengaturan-sistem");
  const navUserManagement = document.getElementById("nav-user-management");

  // Dashboard Section
  const startDateInput = document.getElementById("start-date");
  const endDateInput = document.getElementById("end-date");
  const filterBtn = document.getElementById("filter-btn");
  const totalPendapatanDisplay = document.getElementById("total-pendapatan");
  const totalKendaraanDisplay = document.getElementById("total-kendaraan");
  const avgDurationDisplay = document.getElementById("avg-duration");

  // Monitoring Section
  const btnManualEntry = document.getElementById("btn-manual-entry");
  const kapasitasContainer = document.getElementById("kapasitas-container");
  const recentActivityTableBody = document.querySelector(
    "#recent-activity-table tbody"
  );

  // Monitoring Section - CCTV Elements
  const entryFileInput = document.getElementById("entry-file-input");
  const entryLoader = document.getElementById("entry-loader");
  const entryResultImageContainer = document.getElementById(
    "entry-result-image-container"
  );
  const entryResultText = document.getElementById("entry-result-text");
  const exitFileInput = document.getElementById("exit-file-input");
  const exitLoader = document.getElementById("exit-loader");
  const exitResultImageContainer = document.getElementById(
    "exit-result-image-container"
  );
  const exitResultText = document.getElementById("exit-result-text");
  //Akses Section
  const navManajemenAkses = document.getElementById("nav-manajemen-akses");
  const formAkses = document.getElementById("form-akses");
  const whitelistTableBody = document.getElementById("whitelist-table-body");
  const blacklistTableBody = document.getElementById("blacklist-table-body");

  // Laporan Section
  const laporanStartDateInput = document.getElementById("laporan-start-date");
  const laporanEndDateInput = document.getElementById("laporan-end-date");
  const laporanJenisKendaraanSelect = document.getElementById(
    "laporan-jenis-kendaraan"
  );
  const laporanTipePlatSelect = document.getElementById("laporan-tipe-plat");
  const laporanFilterBtn = document.getElementById("laporan-filter-btn");
  const laporanResetBtn = document.getElementById("laporan-reset-btn");
  const laporanDownloadPdfBtn = document.getElementById(
    "laporan-download-pdf-btn"
  );
  const laporanDownloadCsvBtn = document.getElementById(
    "laporan-download-csv-btn"
  );
  const laporanTableBody = document.getElementById("laporan-table-body");

  // Pengaturan Sistem Section
  const settingNamaLokasi = document.getElementById("setting-nama-lokasi");
  const settingAlamatLokasi = document.getElementById("setting-alamat-lokasi");
  const settingTeleponKontak = document.getElementById(
    "setting-telepon-kontak"
  );
  const settingEmailKontak = document.getElementById("setting-email-kontak");
  const settingMataUang = document.getElementById("setting-mata-uang");
  const settingTarifPertamaMobil = document.getElementById(
    "setting-tarif-pertama-mobil"
  );
  const settingTarifPerjamMobil = document.getElementById(
    "setting-tarif-perjam-mobil"
  );
  const settingTarifMaksimumHarianMobil = document.getElementById(
    "setting-tarif-maksimum-harian-mobil"
  );
  const settingTarifPertamaMotor = document.getElementById(
    "setting-tarif-pertama-motor"
  );
  const settingTarifPerjamMotor = document.getElementById(
    "setting-tarif-perjam-motor"
  );
  const settingTarifMaksimumHarianMotor = document.getElementById(
    "setting-tarif-maksimum-harian-motor"
  );
  const settingToleransiMasukMenit = document.getElementById(
    "setting-toleransi-masuk-menit"
  );
  const settingKapasitasMaksimalMobil = document.getElementById(
    "setting-kapasitas-maksimal-mobil"
  );
  const settingKapasitasMaksimalMotor = document.getElementById(
    "setting-kapasitas-maksimal-motor"
  );
  const simpanPengaturanBtn = document.getElementById("simpan-pengaturan-btn");

  // User Management Section
  const btnAddUser = document.getElementById("btn-add-user");
  const userTableBody = document.getElementById("user-table-body");

  // Modal
  const modalContainer = document.getElementById("modal-container");
  const modalTitle = document.getElementById("modal-title");
  const modalBody = document.getElementById("modal-body");
  const modalCloseBtn = document.querySelector(".modal-close-btn");

  // Notification
  const notificationContainer = document.getElementById(
    "notification-container"
  );

  // --- State Variables ---
  let currentUser = null;
  let charts = {};

  // --- Utility Functions ---
  function showNotification(message, type = "success", duration = 3000) {
    const notification = document.createElement("div");
    notification.className = `notification ${type}`;
    notification.innerHTML = `<i class="fas ${
      type === "success" ? "fa-check-circle" : "fa-times-circle"
    }"></i> ${message}`;
    notificationContainer.appendChild(notification);
    setTimeout(() => {
      notification.classList.add("fade-out");
      setTimeout(() => notification.remove(), 500);
    }, duration);
  }

  function formatCurrency(amount) {
    const mataUang = localStorage.getItem("setting-mata-uang") || "Rp";
    return `${mataUang} ${Number(amount).toLocaleString("id-ID")}`;
  }

  function formatDateTime(dateInput) {
    if (!dateInput) return "N/A";
    try {
      const date = new Date(dateInput);
      if (isNaN(date.getTime())) return "N/A";
      return date
        .toLocaleString("id-ID", {
          year: "numeric",
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
          hour12: false,
        })
        .replace(/\./g, ":");
    } catch (error) {
      console.error("Error formatting date:", dateInput, error);
      return "N/A";
    }
  }

  function setDefaultDates() {
    const today = new Date();
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(today.getDate() - 6);
    const formatDate = (date) => date.toISOString().split("T")[0];
    if (startDateInput) startDateInput.value = formatDate(sevenDaysAgo);
    if (endDateInput) endDateInput.value = formatDate(today);
    if (laporanStartDateInput)
      laporanStartDateInput.value = formatDate(sevenDaysAgo);
    if (laporanEndDateInput) laporanEndDateInput.value = formatDate(today);
  }

  // --- API Call Functions ---
    async function apiCall(url, options = {}) {
        const isFormData = options.body instanceof FormData;
        const defaultHeaders = isFormData ? {} : { 'Content-Type': 'application/json' };

        try {
            const response = await fetch(url, {
                headers: { ...defaultHeaders, ...options.headers },
                ...options,
            });
            if (!response.ok) {
                // Tangkap data error, terutama untuk status 403 (blacklist)
                const errorData = await response.json().catch(() => ({ detail: `HTTP error! status: ${response.status}` }));
                console.error(`API Error (${url}): ${response.status}`, errorData);
                // Kita lempar error agar bisa ditangkap di blok catch pemanggil
                throw new Error(errorData.detail || `Server error: ${response.status}`);
            }
            const contentType = response.headers.get("content-type");
            if (contentType?.includes("application/json")) return await response.json();
            if (contentType?.includes("application/pdf") || contentType?.includes("text/csv")) return response.blob();
            return response.text();
        } catch (error) {
            console.error('API Call Failed:', url, error);
            // Tampilkan notifikasi error ke pengguna
            showNotification(error.message || 'Terjadi kesalahan jaringan.', 'error');
            throw error; // Lempar kembali error agar proses selanjutnya berhenti
        }
    }

  // --- Chart Functions ---
  function renderChart(chartId, chartType, data, options) {
    const ctx = document.getElementById(chartId);
    if (!ctx) {
      console.error(`Canvas element with ID ${chartId} not found.`);
      return;
    }
    if (charts[chartId]) {
      charts[chartId].destroy();
    }
    charts[chartId] = new Chart(ctx.getContext("2d"), {
      type: chartType,
      data,
      options,
    });
  }

  // --- CCTV FUNCTIONS ---
  function setupCCTVUpload(config) {
    const fileInput = document.getElementById(config.inputId);
    const loader = document.getElementById(config.loaderId);
    const imageContainer = document.getElementById(config.imageContainerId);
    const resultText = document.getElementById(config.textResultId);

    if (!fileInput) return;

    fileInput.addEventListener("change", async function (event) {
      const file = event.target.files[0];
      if (!file) return;

      loader.style.display = "block";
      resultText.innerHTML = "";
      imageContainer.innerHTML = `<div class="cctv-placeholder"><i class="fas fa-sync-alt fa-spin"></i><span>Memproses...</span></div>`;

      const formData = new FormData();
      formData.append("file", file);

      try {
        const data = await apiCall(config.apiUrl, {
          method: "POST",
          body: formData,
        });
        config.displayFunction(data, imageContainer, resultText);
        if (typeof loadMonitoringData === "function") {
          loadMonitoringData();
        }
      } catch (error) {
        resultText.innerHTML = `<p class="status-error"><strong>Gagal Menganalisis:</strong> ${error.message}</p>`;
        imageContainer.innerHTML = `<div class="cctv-placeholder"><i class="fas fa-times-circle"></i><span>Analisis Gagal</span></div>`;
      } finally {
        loader.style.display = "none";
        fileInput.value = "";
      }
    });
  }

  // GANTI SELURUH FUNGSI LAMA ANDA DENGAN YANG INI
  function displayEntryResults(data, imageContainer, resultText) {
    if (data.annotated_image) {
      imageContainer.innerHTML = `<img src="${data.annotated_image}" alt="Hasil Deteksi" class="result-image">`;
    } else {
      imageContainer.innerHTML = `<div class="cctv-placeholder"><i class="fas fa-image"></i><span>Tidak ada gambar</span></div>`;
    }

    // Untuk Debugging: Anda bisa aktifkan baris ini untuk melihat data dari server
    // console.log("Data Masuk:", JSON.stringify(data, null, 2));

    let html = '<h5><i class="fas fa-poll"></i> Hasil Analisis Masuk</h5>';

    // 1. Menampilkan Nomor Plat
    if (
      data.ocr_results &&
      data.ocr_results.length > 0 &&
      data.ocr_results[0].status === "success"
    ) {
      html += `<p><strong>Nomor Plat:</strong> <span>${data.ocr_results[0].text}</span></p>`;
    } else {
      html += `<p><strong>Nomor Plat:</strong> <span class="status-error">Tidak terbaca</span></p>`;
    }

    // 2. Menampilkan Jenis Kendaraan
    if (data.vehicle_detections && data.vehicle_detections.results.length > 0) {
      const vehicle = data.vehicle_detections.results.find(
        (d) => !d.class_name.toLowerCase().includes("plat")
      );
      html += `<p><strong>Jenis Kendaraan:</strong> <span>${
        vehicle ? vehicle.class_name : "Tidak Diketahui"
      }</span></p>`;
    } else {
      html += `<p><strong>Jenis Kendaraan:</strong> <span class="status-info">Tidak terdeteksi</span></p>`;
    }

    // =======================================================
    // BAGIAN YANG DITAMBAHKAN UNTUK MENAMPILKAN TIPE PLAT
    // =======================================================
    // 3. Menampilkan Tipe Plat
    if (
      data.type_plate_detections &&
      data.type_plate_detections.results.length > 0
    ) {
      const plateType = data.type_plate_detections.results[0].class_name;
      html += `<p><strong>Tipe Plat:</strong> <span>${plateType}</span></p>`;
    } else {
      html += `<p><strong>Tipe Plat:</strong> <span class="status-info">Tidak terdeteksi</span></p>`;
    }
    // =======================================================

    // 4. Menampilkan Status Database
    if (data.database_status) {
      const statusClass =
        data.database_status.status === "success"
          ? "status-success"
          : "status-error";
      html += `<p><strong>Status DB:</strong> <span class="${statusClass}">${data.database_status.message}</span></p>`;
    }

    resultText.innerHTML = html;
  }

  function displayExitResults(data, imageContainer, resultText) {
    // --- PERBAIKAN DIMULAI DI SINI ---

    // Prioritaskan untuk menampilkan gambar hasil anotasi dari server
    if (data.annotated_image) {
      imageContainer.innerHTML = `<img src="${data.annotated_image}" alt="Hasil Deteksi Keluar" class="result-image">`;
    } else {
      // Jika karena suatu alasan tidak ada gambar anotasi, tampilkan gambar asli sebagai fallback
      const originalFile = exitFileInput.files[0];
      if (originalFile) {
        imageContainer.innerHTML = `<img src="${URL.createObjectURL(
          originalFile
        )}" alt="Gambar Asli Keluar" class="result-image">`;
      } else {
        // Placeholder jika tidak ada gambar sama sekali
        imageContainer.innerHTML = `<div class="cctv-placeholder"><i class="fas fa-image"></i><span>Tidak ada gambar</span></div>`;
      }
    }

    // --- AKHIR DARI PERBAIKAN ---

    // Sisa dari fungsi (logika untuk menampilkan teks detail) tetap sama
    let html =
      '<h5><i class="fas fa-receipt"></i> Detail Kendaraan Keluar</h5>';
    if (data.exit_details && data.exit_details.success) {
      const details = data.exit_details.detail_parkir;
      const biaya = new Intl.NumberFormat("id-ID", {
        style: "currency",
        currency: "IDR",
        minimumFractionDigits: 0,
      }).format(details.biaya_rp);
      html += `<p><strong>Nomor Plat:</strong> ${details.nomor_plat}</p>`;
      html += `<p><strong>Waktu Masuk:</strong> ${formatDateTime(
        details.waktu_masuk
      )}</p>`;
      html += `<p><strong>Waktu Keluar:</strong> ${formatDateTime(
        details.tanggal_keluar
      )}</p>`;
      html += `<p><strong>Durasi:</strong> ${details.durasi_menit} menit</p>`;
      html += `<p><strong>Total Biaya:</strong> <strong>${biaya}</strong></p>`;
      html += `<p class="status-success">${data.exit_details.message}</p>`;
    } else {
      html += `<p class="status-error"><strong>Error:</strong> ${
        data.detail || data.message || "Terjadi kesalahan."
      }</p>`;
    }
    resultText.innerHTML = html;
  }

  // --- Page Load and Update Functions ---
  async function loadDashboardStats() {
    const startDate = startDateInput.value;
    const endDate = endDateInput.value;
    if (!startDate || !endDate) {
      showNotification("Silakan pilih rentang tanggal.", "error");
      return;
    }
    try {
      const data = await apiCall(
        `/api/dashboard_stats?start_date=${startDate}&end_date=${endDate}`
      );
      if (data.stat_cards) {
        updateDashboardUI(
          data.stat_cards,
          data.pendapatan_harian,
          data.arus_per_jam,
          data.distribusi_kendaraan,
          data.pendapatan_per_jenis
        );
      } else {
        updateDashboardUI(
          { total_pendapatan: 0, total_kendaraan: 0, avg_duration_minutes: 0 },
          [],
          [],
          [],
          []
        );
      }
    } catch (error) {
      updateDashboardUI(
        { total_pendapatan: 0, total_kendaraan: 0, avg_duration_minutes: 0 },
        [],
        [],
        [],
        []
      );
    }
  }

  function updateDashboardUI(
    statCards,
    pendapatanHarian,
    arusPerJam,
    distribusiKendaraan,
    pendapatanPerJenis
  ) {
    totalPendapatanDisplay.textContent = formatCurrency(
      statCards.total_pendapatan || 0
    );
    totalKendaraanDisplay.textContent = statCards.total_kendaraan || 0;
    avgDurationDisplay.textContent = `${Math.round(
      statCards.avg_duration_minutes || 0
    )} Menit`;
    renderChart(
      "pendapatanHarianChart",
      "line",
      {
        labels: pendapatanHarian.map((item) => item.tanggal),
        datasets: [
          {
            label: "Pendapatan Harian",
            data: pendapatanHarian.map((item) => item.pendapatan),
            borderColor: "rgb(75, 192, 192)",
            tension: 0.1,
          },
        ],
      },
      {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { title: { display: true, text: "Pendapatan Harian" } },
      }
    );
    renderChart(
      "arusPerJamChart",
      "bar",
      {
        labels: arusPerJam.map((item) => `${item.jam}:00`),
        datasets: [
          {
            label: "Jumlah Kendaraan Masuk",
            data: arusPerJam.map((item) => item.jumlah),
            backgroundColor: "rgba(255, 159, 64, 0.7)",
          },
        ],
      },
      {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: "Arus Kendaraan per Jam (Masuk)" },
        },
      }
    );
    renderChart(
      "distribusiKendaraanChart",
      "pie",
      {
        labels: distribusiKendaraan.map((item) => item.jenis_kendaraan),
        datasets: [
          {
            data: distribusiKendaraan.map((item) => item.jumlah),
            backgroundColor: [
              "rgba(54, 162, 235, 0.7)",
              "rgba(255, 206, 86, 0.7)",
            ],
          },
        ],
      },
      {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: "Distribusi Jenis Kendaraan (Masuk)" },
        },
      }
    );
    renderChart(
      "pendapatanPerJenisChart",
      "doughnut",
      {
        labels: pendapatanPerJenis.map((item) => item.jenis_kendaraan),
        datasets: [
          {
            data: pendapatanPerJenis.map((item) => item.total_pendapatan),
            backgroundColor: [
              "rgba(153, 102, 255, 0.7)",
              "rgba(255, 99, 132, 0.7)",
            ],
          },
        ],
      },
      {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: "Pendapatan per Jenis Kendaraan" },
        },
      }
    );
  }

  async function loadMonitoringData() {
    try {
      const data = await apiCall("/api/monitoring");
      if (kapasitasContainer) {
        kapasitasContainer.innerHTML = "";
        if (data.kapasitas && data.kapasitas.length > 0) {
          data.kapasitas.forEach((item) => {
            const card = document.createElement("div");
            card.className = "card capacity-card";
            const terisi = Number(item.kapasitas_terisi) || 0;
            const total = Number(item.kapasitas_total) || 0;
            const persen = total > 0 ? (terisi / total) * 100 : 0;
            let barClass = "progress-bar";
            if (persen > 90) barClass += " over-capacity";
            else if (persen > 70) barClass += " full-capacity";
            card.innerHTML = `<h4><i class="fas ${
              item.jenis_kendaraan.toLowerCase() === "mobil"
                ? "fa-car-side"
                : "fa-motorcycle"
            }"></i> ${
              item.jenis_kendaraan
            }</h4><p>${terisi} / ${total}</p><div class="progress-bar-container"><div class="${barClass}" style="width: ${Math.min(
              persen,
              100
            ).toFixed(2)}%;">${persen.toFixed(0)}%</div></div>`;
            kapasitasContainer.appendChild(card);
          });
        } else {
          kapasitasContainer.innerHTML =
            "<p>Data kapasitas tidak tersedia.</p>";
        }
      }
      if (recentActivityTableBody) {
        recentActivityTableBody.innerHTML = "";
        if (data.recent_activities && data.recent_activities.length > 0) {
          data.recent_activities.forEach((activity) => {
            const row = recentActivityTableBody.insertRow();
            row.innerHTML = `<td>${activity.nomor_plat || "N/A"}</td><td>${
              activity.jenis_kendaraan || "N/A"
            }</td><td>${activity.tipe_plat || "Sipil"}</td><td>${formatDateTime(
              activity.tanggal_masuk
            )}</td><td><span class="status status-${activity.status.toLowerCase()}">${
              activity.status
            }</span></td>`;
          });
        } else {
          recentActivityTableBody.innerHTML =
            '<tr><td colspan="5">Tidak ada aktivitas terkini.</td></tr>';
        }
      }
    } catch (error) {
      if (kapasitasContainer)
        kapasitasContainer.innerHTML = "<p>Gagal memuat data kapasitas.</p>";
      if (recentActivityTableBody)
        recentActivityTableBody.innerHTML =
          '<tr><td colspan="5">Gagal memuat aktivitas.</td></tr>';
    }
  }

  async function loadLaporanData() {
    const params = new URLSearchParams({
      start_date: laporanStartDateInput.value,
      end_date: laporanEndDateInput.value,
      jenis_kendaraan: laporanJenisKendaraanSelect.value,
      tipe_plat: laporanTipePlatSelect.value,
    });
    try {
      const data = await apiCall(`/api/laporan?${params.toString()}`);
      laporanTableBody.innerHTML = "";
      if (data.laporan_data && data.laporan_data.length > 0) {
        data.laporan_data.forEach((item) => {
          const row = laporanTableBody.insertRow();
          row.innerHTML = `<td>${item.nomor_plat || "N/A"}</td><td>${
            item.jenis_kendaraan || "N/A"
          }</td><td>${item.tipe_plat || "Sipil"}</td><td>${formatDateTime(
            item.tanggal_masuk
          )}</td><td>${formatDateTime(item.tanggal_keluar)}</td><td>${
            item.durasi || 0
          }</td><td>${formatCurrency(item.biaya || 0)}</td>`;
        });
      } else {
        laporanTableBody.innerHTML =
          '<tr><td colspan="7">Tidak ada data laporan untuk filter yang dipilih.</td></tr>';
      }
    } catch (error) {
      laporanTableBody.innerHTML =
        '<tr><td colspan="7">Gagal memuat data laporan.</td></tr>';
    }
  }

  async function loadPengaturanSistem() {
    try {
      const s = await apiCall("/api/pengaturan");
      if (s) {
        settingNamaLokasi.value = s.nama_lokasi_parkir || "";
        settingAlamatLokasi.value = s.alamat_lokasi_parkir || "";
        settingTeleponKontak.value = s.telepon_kontak_parkir || "";
        settingEmailKontak.value = s.email_kontak_parkir || "";
        settingMataUang.value = s.simbol_mata_uang || "Rp";
        settingTarifPertamaMobil.value = s.tarif_pertama_mobil || "";
        settingTarifPerjamMobil.value = s.tarif_perjam_mobil || "";
        settingTarifMaksimumHarianMobil.value =
          s.tarif_maksimum_harian_mobil || "";
        settingTarifPertamaMotor.value = s.tarif_pertama_motor || "";
        settingTarifPerjamMotor.value = s.tarif_perjam_motor || "";
        settingTarifMaksimumHarianMotor.value =
          s.tarif_maksimum_harian_motor || "";
        settingToleransiMasukMenit.value = s.toleransi_masuk_menit || "";
        // Kapasitas dimuat dari API monitoring, jadi tidak di-set di sini.
      }
    } catch (error) {
      showNotification("Gagal memuat pengaturan.", "error");
    }
  }

  async function loadUserManagement() {
    try {
      const users = await apiCall("/api/users");
      userTableBody.innerHTML = "";
      if (users && users.length > 0) {
        users.forEach((user) => {
          const row = userTableBody.insertRow();
          row.innerHTML = `<td>${user.username}</td><td>${user.role}</td><td><button class="btn-action btn-edit-user" data-id="${user.id_user}" data-username="${user.username}" data-role="${user.role}"><i class="fas fa-edit"></i></button><button class="btn-action btn-delete-user" data-id="${user.id_user}"><i class="fas fa-trash-alt"></i></button></td>`;
        });
        document
          .querySelectorAll(".btn-edit-user")
          .forEach((btn) => btn.addEventListener("click", handleEditUserModal));
        document
          .querySelectorAll(".btn-delete-user")
          .forEach((btn) => btn.addEventListener("click", handleDeleteUser));
      } else {
        userTableBody.innerHTML =
          '<tr><td colspan="3">Tidak ada pengguna terdaftar.</td></tr>';
      }
    } catch (error) {
      userTableBody.innerHTML =
        '<tr><td colspan="3">Gagal memuat data pengguna.</td></tr>';
    }
  }

  // --- Modal Functions ---
  function openModal(title, contentHTML, submitCallback) {
    modalTitle.textContent = title;
    modalBody.innerHTML = contentHTML;
    modalContainer.classList.remove("hidden");
    const formInModal = modalBody.querySelector("form");
    if (formInModal && submitCallback) {
      formInModal.onsubmit = async (e) => {
        e.preventDefault();
        await submitCallback(new FormData(formInModal));
      };
    }
  }

  function closeModal() {
    modalContainer.classList.add("hidden");
    modalBody.innerHTML = "";
  }

  // --- Event Handlers ---
  if (loginForm) {
    loginForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      loginError.textContent = "";
      const username = usernameInput.value;
      const password = passwordInput.value;
      try {
        const data = await apiCall("/api/login", {
          method: "POST",
          body: JSON.stringify({ username, password }),
        });
        if (data.success) {
          currentUser = data.user;
          localStorage.setItem("currentUser", JSON.stringify(currentUser));
          localStorage.setItem("lastActivePage", "dashboard");
          setupAppUI();
          loginPage.classList.add("hidden");
          appContainer.classList.remove("hidden");
          navigateTo("dashboard");
          showNotification(`Selamat datang, ${currentUser.name}!`, "success");
        } else {
          loginError.textContent = data.detail.error || "Login gagal.";
        }
      } catch (error) {
        loginError.textContent = "Username atau password salah.";
      }
    });
  }

  function handleLogout() {
    currentUser = null;
    localStorage.removeItem("currentUser");
    localStorage.removeItem("lastActivePage");
    if (appContainer) appContainer.classList.add("hidden");
    if (loginPage) loginPage.classList.remove("hidden");
    if (usernameInput) usernameInput.value = "";
    if (passwordInput) passwordInput.value = "";
    if (loginError) loginError.textContent = "";
    showNotification("Anda telah logout.", "success");
  }

  if (logoutBtn) logoutBtn.addEventListener("click", handleLogout);

  navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      navigateTo(link.dataset.page);
    });
  });

  if (filterBtn) filterBtn.addEventListener("click", loadDashboardStats);

  if (btnManualEntry) {
    btnManualEntry.addEventListener("click", () => {
      const now = new Date().toISOString().slice(0, 16);
      openModal(
        "Entri Kendaraan Manual",
        `
                <form><div class="form-group"><label>Nomor Plat:</label><input type="text" name="nomor_plat" required></div><div class="form-group"><label>Jenis Kendaraan:</label><select name="jenis_kendaraan" required><option value="Mobil">Mobil</option><option value="Motor">Motor</option></select></div><div class="form-group"><label>Tipe Plat:</label><select name="tipe_plat"><option value="Sipil">Sipil</option><option value="Pemerintah">Pemerintah</option><option value="TNI AD">TNI AD</option></select></div><div class="form-group"><label>Tanggal Masuk:</label><input type="datetime-local" name="tanggal_masuk" value="${now}" required></div><button type="submit" class="btn-primary">Simpan</button></form>
            `,
        async (formData) => {
          const entryData = {
            nomor_plat: formData.get("nomor_plat"),
            jenis_kendaraan: formData.get("jenis_kendaraan"),
            tipe_plat: formData.get("tipe_plat"),
            tanggal_masuk: new Date(
              formData.get("tanggal_masuk")
            ).toISOString(),
          };
          try {
            await apiCall("/api/kendaraan/masuk", {
              method: "POST",
              body: JSON.stringify(entryData),
            });
            showNotification("Entri manual berhasil.", "success");
            loadMonitoringData();
            closeModal();
          } catch (error) {
            showNotification(error.message, "error");
          }
        }
      );
    });
  }

  if (laporanFilterBtn)
    laporanFilterBtn.addEventListener("click", loadLaporanData);
  if (laporanResetBtn) {
    laporanResetBtn.addEventListener("click", () => {
      setDefaultDates();
      if (laporanJenisKendaraanSelect) laporanJenisKendaraanSelect.value = "";
      if (laporanTipePlatSelect) laporanTipePlatSelect.value = "";
      loadLaporanData();
    });
  }

  async function downloadReport(format) {
    const params = new URLSearchParams({
      start_date: laporanStartDateInput.value,
      end_date: laporanEndDateInput.value,
      jenis_kendaraan: laporanJenisKendaraanSelect.value,
      tipe_plat: laporanTipePlatSelect.value,
    });
    try {
      const blob = await apiCall(
        `/api/laporan/${format}?${params.toString()}`,
        { method: "GET" }
      );
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `laporan_${
        new Date().toISOString().split("T")[0]
      }.${format}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      /* notif handled by apiCall */
    }
  }

  if (laporanDownloadPdfBtn)
    laporanDownloadPdfBtn.addEventListener("click", () =>
      downloadReport("pdf")
    );
  if (laporanDownloadCsvBtn)
    laporanDownloadCsvBtn.addEventListener("click", () =>
      downloadReport("csv")
    );

  if (simpanPengaturanBtn) {
    simpanPengaturanBtn.addEventListener("click", async () => {
      const settings = {
        nama_lokasi_parkir: settingNamaLokasi.value,
        alamat_lokasi_parkir: settingAlamatLokasi.value,
        //... add all other settings
      };
      try {
        await apiCall("/api/pengaturan", {
          method: "POST",
          body: JSON.stringify(settings),
        });
        // update kapasitas terpisah
        await apiCall("/api/kapasitas/update", {
          method: "POST",
          body: JSON.stringify({
            jenis_kendaraan: "Mobil",
            kapasitas_total: parseInt(settingKapasitasMaksimalMobil.value),
          }),
        });
        await apiCall("/api/kapasitas/update", {
          method: "POST",
          body: JSON.stringify({
            jenis_kendaraan: "Motor",
            kapasitas_total: parseInt(settingKapasitasMaksimalMotor.value),
          }),
        });
        showNotification("Pengaturan disimpan.", "success");
      } catch (error) {
        /* notif handled by apiCall */
      }
    });
  }

  if (btnAddUser) {
    btnAddUser.addEventListener("click", () => {
      openModal(
        "Tambah Pengguna",
        `<form><div class="form-group"><label>Username:</label><input type="text" name="username" required></div><div class="form-group"><label>Password:</label><input type="password" name="password" required></div><div class="form-group"><label>Role:</label><select name="role"><option value="operator">Operator</option><option value="admin">Admin</option></select></div><button type="submit" class="btn-primary">Simpan</button></form>`,
        async (formData) => {
          const data = {
            username: formData.get("username"),
            password: formData.get("password"),
            role: formData.get("role"),
          };
          try {
            await apiCall("/api/users", {
              method: "POST",
              body: JSON.stringify(data),
            });
            showNotification("Pengguna ditambahkan.", "success");
            loadUserManagement();
            closeModal();
          } catch (error) {
            /* notif handled by apiCall */
          }
        }
      );
    });
  }

  function handleEditUserModal(e) {
    const { id, username, role } = e.currentTarget.dataset;
    openModal(
      "Edit Pengguna",
      `<form><div class="form-group"><label>Username:</label><input type="text" name="username" value="${username}" required></div><div class="form-group"><label>Password (baru):</label><input type="password" name="password" placeholder="Kosongkan jika tidak diubah"></div><div class="form-group"><label>Role:</label><select name="role"><option value="operator" ${
        role === "operator" ? "selected" : ""
      }>Operator</option><option value="admin" ${
        role === "admin" ? "selected" : ""
      }>Admin</option></select></div><button type="submit" class="btn-primary">Update</button></form>`,
      async (formData) => {
        const data = {
          username: formData.get("username"),
          role: formData.get("role"),
          password: formData.get("password"),
        };
        if (!data.password) delete data.password;
        try {
          await apiCall(`/api/users/${id}`, {
            method: "PUT",
            body: JSON.stringify(data),
          });
          showNotification("Pengguna diupdate.", "success");
          loadUserManagement();
          closeModal();
        } catch (error) {
          /* notif handled by apiCall */
        }
      }
    );
  }

  async function handleDeleteUser(e) {
    const { id } = e.currentTarget.dataset;
    if (confirm("Yakin ingin menghapus pengguna ini?")) {
      try {
        await apiCall(`/api/users/${id}`, { method: "DELETE" });
        showNotification("Pengguna dihapus.", "success");
        loadUserManagement();
      } catch (error) {
        /* notif handled by apiCall */
      }
    }
  }

  if (modalCloseBtn) modalCloseBtn.addEventListener("click", closeModal);
  if (modalContainer) {
    modalContainer.addEventListener("click", (e) => {
      if (e.target === modalContainer) closeModal();
    });
  }

   async function loadAksesData() {
        if (!whitelistTableBody || !blacklistTableBody) return;
        try {
            const dataAkses = await apiCall('/api/akses');
            
            // Kosongkan tabel sebelum diisi ulang
            whitelistTableBody.innerHTML = '<tr><td colspan="2">Memuat...</td></tr>';
            blacklistTableBody.innerHTML = '<tr><td colspan="2">Memuat...</td></tr>';
            
            let whitelistHTML = '';
            let blacklistHTML = '';

            dataAkses.forEach(item => {
                const rowHTML = `
                    <tr>
                        <td>${item.nomor_plat}</td>
                        <td class="actions-cell">
                            <button class="btn-action btn-delete-akses" data-id="${item.id}" title="Hapus">
                                <i class="fas fa-trash-alt"></i>
                            </button>
                        </td>
                    </tr>
                `;
                if (item.status === 'WHITELIST') {
                    whitelistHTML += rowHTML;
                } else {
                    blacklistHTML += rowHTML;
                }
            });

            whitelistTableBody.innerHTML = whitelistHTML || '<tr><td colspan="2">Tidak ada data.</td></tr>';
            blacklistTableBody.innerHTML = blacklistHTML || '<tr><td colspan="2">Tidak ada data.</td></tr>';

            // Tambahkan event listener untuk tombol hapus yang baru dibuat
            document.querySelectorAll('.btn-delete-akses').forEach(btn => {
                btn.addEventListener('click', handleDeleteAkses);
            });

        } catch (error) {
            whitelistTableBody.innerHTML = '<tr><td colspan="2">Gagal memuat data.</td></tr>';
            blacklistTableBody.innerHTML = '<tr><td colspan="2">Gagal memuat data.</td></tr>';
        }
    }

    async function handleAddAkses(event) {
        event.preventDefault();
        const nomorPlatInput = document.getElementById('akses-nomor-plat');
        const statusSelect = document.getElementById('akses-status');
        
        const data = {
            nomor_plat: nomorPlatInput.value.trim().toUpperCase(),
            status: statusSelect.value
        };

        if (!data.nomor_plat) {
            showNotification('Nomor plat tidak boleh kosong.', 'error');
            return;
        }

        try {
            const result = await apiCall('/api/akses', {
                method: 'POST',
                body: JSON.stringify(data)
            });
            showNotification(result.message, 'success');
            formAkses.reset();
            loadAksesData(); // Muat ulang data setelah berhasil
        } catch (error) {
            // Notifikasi error sudah ditangani oleh apiCall
        }
    }

    async function handleDeleteAkses(event) {
        const id = event.currentTarget.dataset.id;
        if (confirm('Anda yakin ingin menghapus plat ini dari daftar?')) {
            try {
                const result = await apiCall(`/api/akses/${id}`, { method: 'DELETE' });
                showNotification(result.message, 'success');
                loadAksesData(); // Muat ulang data setelah berhasil
            } catch (error) {
                // Notifikasi error sudah ditangani oleh apiCall
            }
        }
    }
  // --- Initial Setup & Navigation ---
  function navigateTo(page) {
    navLinks.forEach((l) => l.classList.remove("active"));
    const activeLink = document.querySelector(`.nav-link[data-page="${page}"]`);
    if (activeLink) activeLink.classList.add("active");

    if (pageTitle)
      pageTitle.textContent = activeLink
        ? activeLink.querySelector("span").textContent
        : "Smart Parking";
    pageSections.forEach((section) => section.classList.add("hidden"));
    const activeSection = document.getElementById(`${page}-section`);
    if (activeSection) activeSection.classList.remove("hidden");

    try {
      localStorage.setItem("lastActivePage", page);
    } catch (e) {
      console.error(e);
    }

    if (page === "dashboard") loadDashboardStats();
    if (page === "monitoring") loadMonitoringData();
    if (page === "laporan") loadLaporanData();
    if (page === "pengaturan-sistem" && currentUser?.role === "admin")
      loadPengaturanSistem();
    if (page === "user-management" && currentUser?.role === "admin")
      loadUserManagement();
    if (page === 'manajemen-akses' && currentUser?.role === 'admin') loadAksesData();
  }

  function setupAppUI() {
    if (currentUser) {
      if (userNameDisplay) userNameDisplay.textContent = currentUser.name;
      if (userAvatarDisplay)
        userAvatarDisplay.textContent =
          currentUser.avatar || currentUser.name.charAt(0).toUpperCase();
      if (currentUser.role === "admin") {
        if (navPengaturanSistem) navPengaturanSistem.classList.remove("hidden");
        if (navUserManagement) navUserManagement.classList.remove("hidden");
      } else {
        if (navPengaturanSistem) navPengaturanSistem.classList.add("hidden");
        if (navUserManagement) navUserManagement.classList.add("hidden");
      }
      if (navManajemenAkses) navManajemenAkses.classList.remove('hidden');
            } else {
    }
  }

  function init() {
    setDefaultDates();
    const storedUser = localStorage.getItem("currentUser");
    if (storedUser) {
      currentUser = JSON.parse(storedUser);
      setupAppUI();
      if (loginPage) loginPage.classList.add("hidden");
      if (appContainer) appContainer.classList.remove("hidden");
      navigateTo(localStorage.getItem("lastActivePage") || "dashboard");
    } else {
      if (loginPage) loginPage.classList.remove("hidden");
      if (appContainer) appContainer.classList.add("hidden");
    }
    if (formAkses) {
            formAkses.addEventListener('submit', handleAddAkses);
        }

    // Setup listeners untuk CCTV
    setupCCTVUpload({
      inputId: "entry-file-input",
      loaderId: "entry-loader",
      imageContainerId: "entry-result-image-container",
      textResultId: "entry-result-text",
      apiUrl: "/api/detect/entry",
      displayFunction: displayEntryResults,
    });
    setupCCTVUpload({
      inputId: "exit-file-input",
      loaderId: "exit-loader",
      imageContainerId: "exit-result-image-container",
      textResultId: "exit-result-text",
      apiUrl: "/api/detect/exit",
      displayFunction: displayExitResults,
    });
  }

  init();
});
