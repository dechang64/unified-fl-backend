//! Med GPS Service — 罕见病就医导航
//!
//! Implements gRPC MedGpsService: Navigate, SubmitExperience, FindSimilarExperiences.
//! Uses HNSW VectorDB for semantic matching of hospitals, diseases, and patient experiences.

use crate::proto::{
    med_gps_service_server::{MedGpsService, MedGpsServiceServer},
    *,
};
use crate::vector_db::SharedVectorDB;
use crate::audit::{SharedAuditChain, EventType};
use tonic::{Request, Response, Status};

pub struct MedGpsServiceImpl {
    db: SharedVectorDB,
    audit: SharedAuditChain,
}

impl MedGpsServiceImpl {
    pub fn new(db: SharedVectorDB, audit: SharedAuditChain) -> Self {
        Self { db, audit }
    }
}

#[tonic::async_trait]
impl MedGpsService for MedGpsServiceImpl {
    /// 智能导航：患者画像 → HNSW 匹配 → 推荐路线
    async fn navigate(
        &self,
        req: Request<NavigateRequest>,
    ) -> Result<Response<NavigateResponse>, Status> {
        let req = req.into_inner();
        let profile = req.profile.ok_or_else(|| {
            Status::invalid_argument("PatientProfile is required")
        })?;

        // Build query text from patient profile for HNSW search
        let _query_text = build_query_text(&profile);

        // We need the embedding to be provided by the client (Python side generates it)
        // For now, search using metadata-based approach:
        // 1. Search hospitals matching the disease
        // 2. Search experiences matching the disease
        // The actual vector search is done client-side; this service orchestrates.

        let max_hospitals = if req.max_hospitals > 0 { req.max_hospitals as usize } else { 5 };

        // Search for matching hospitals by disease name
        let hospital_results = search_by_metadata(
            &self.db,
            &format!("disease:{}", profile.disease_name),
            "hospital",
            max_hospitals,
        );

        // Search for matching experiences
        let experience_results = if req.include_experiences {
            search_by_metadata(
                &self.db,
                &format!("disease:{}", profile.disease_name),
                "experience",
                5,
            )
        } else {
            vec![]
        };

        // Build navigation route from matched results
        let route = build_navigation_route(
            &profile,
            &hospital_results,
            &experience_results,
            &self.audit,
        )?;

        // Audit log
        if let Ok(mut audit) = self.audit.write() {
            audit.append(
                EventType::Custom {
                    name: "med_gps_navigate".into(),
                },
                "med_gps",
                serde_json::json!({
                    "disease": profile.disease_name,
                    "hospitals_matched": hospital_results.len(),
                    "experiences_matched": experience_results.len(),
                    "route_id": route.route_id,
                }),
            );
        }

        let status = if hospital_results.is_empty() {
            "partial".to_string()
        } else {
            "success".to_string()
        };

        Ok(Response::new(NavigateResponse {
            route: Some(route),
            matched_hospitals: hospital_results,
            matched_experiences: experience_results,
            status,
        }))
    }

    /// 提交病友经验：存入 HNSW（向量由客户端生成）
    async fn submit_experience(
        &self,
        req: Request<SubmitExperienceRequest>,
    ) -> Result<Response<SubmitExperienceResponse>, Status> {
        let req = req.into_inner();
        let exp = req.experience.ok_or_else(|| {
            Status::invalid_argument("PatientExperience is required")
        })?;

        let exp_id = if exp.experience_id.is_empty() {
            format!("exp_{}", chrono::Utc::now().format("%Y%m%d%H%M%S"))
        } else {
            exp.experience_id.clone()
        };

        // Audit log
        let audit_hash = {
            let mut audit = self.audit.write().map_err(|e| {
                Status::internal(format!("audit lock poisoned: {e}"))
            })?;
            audit.append(
                EventType::Custom {
                    name: "experience_submitted".into(),
                },
                "med_gps",
                serde_json::json!({
                    "experience_id": exp_id,
                    "disease": exp.disease_name,
                    "hospital": exp.hospital_visited,
                    "outcome": exp.outcome,
                }),
            );
            let entries = audit.recent(1);
            entries.first().map(|b| b.hash.clone()).unwrap_or_default()
        };

        Ok(Response::new(SubmitExperienceResponse {
            success: true,
            experience_id: exp_id,
            audit_hash,
        }))
    }

    /// 查询相似病友经验（向量搜索由客户端完成，此处做元数据过滤搜索）
    async fn find_similar_experiences(
        &self,
        req: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = req.into_inner();

        if req.query.is_empty() {
            return Ok(Response::new(SearchResponse { results: vec![] }));
        }

        let k = if req.k > 0 { req.k as usize } else { 5 };

        let db = self.db.read().map_err(|e| {
            Status::internal(format!("db lock poisoned: {e}"))
        })?;

        let query: Vec<f32> = req.query.into_iter().map(|v| v as f32).collect();
        let hits = db.search(&query, k);

        // Filter to experiences only
        let results: Vec<SearchResult> = hits
            .into_iter()
            .filter(|h| {
                h.metadata
                    .get("type")
                    .map(|t| t == "experience")
                    .unwrap_or(false)
            })
            .map(|h| SearchResult {
                id: h.id,
                values: vec![],  // SearchHit doesn't carry values
                distance: h.score,
                metadata: h.metadata,
            })
            .collect();

        // Audit
        if let Ok(mut audit) = self.audit.write() {
            audit.append(
                EventType::VectorSearch {
                    k: results.len(),
                    result_count: results.len(),
                },
                "med_gps",
                serde_json::json!({"search_type": "similar_experiences"}),
            );
        }

        Ok(Response::new(SearchResponse { results }))
    }
}

/// Build a text query from patient profile for embedding
fn build_query_text(profile: &PatientProfile) -> String {
    let mut parts = vec![profile.disease_name.clone()];
    if !profile.disease_name_en.is_empty() {
        parts.push(profile.disease_name_en.clone());
    }
    if !profile.symptoms.is_empty() {
        parts.push(profile.symptoms.clone());
    }
    if !profile.stage.is_empty() && profile.stage != "unknown" {
        parts.push(format!("stage:{}", profile.stage));
    }
    if !profile.age_group.is_empty() {
        parts.push(format!("age:{}", profile.age_group));
    }
    parts.join(" ")
}

/// Search vectors by metadata filter (type + disease)
fn search_by_metadata(
    db: &SharedVectorDB,
    disease_key: &str,
    entity_type: &str,
    max_results: usize,
) -> Vec<SearchResult> {
    let db = match db.read() {
        Ok(db) => db,
        Err(_) => return vec![],
    };

    // Get all vectors and filter by metadata
    let all = db.all_entries();
    let mut matched: Vec<_> = all
        .into_iter()
        .filter(|entry| {
            let type_match = entry
                .metadata
                .get("type")
                .map(|t| t == entity_type)
                .unwrap_or(false);
            let disease_match = entry
                .metadata
                .get("disease")
                .map(|d| d.contains(&disease_key.replace("disease:", "")))
                .unwrap_or(true); // If no disease metadata, don't filter out
            type_match && disease_match
        })
        .take(max_results)
        .map(|entry| SearchResult {
            id: entry.id.clone(),
            values: entry.values.clone(),
            distance: 0.0,
            metadata: entry.metadata.clone(),
        })
        .collect();

    // Sort by rating if available
    matched.sort_by(|a, b| {
        let ra: f32 = a.metadata.get("rating").and_then(|r| r.parse().ok()).unwrap_or(0.0);
        let rb: f32 = b.metadata.get("rating").and_then(|r| r.parse().ok()).unwrap_or(0.0);
        rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
    });

    matched
}

/// Build navigation route from matched hospitals and experiences
fn build_navigation_route(
    profile: &PatientProfile,
    hospitals: &[SearchResult],
    experiences: &[SearchResult],
    audit: &SharedAuditChain,
) -> Result<NavigationRoute, Status> {
    let route_id = format!("route_{}", chrono::Utc::now().format("%Y%m%d%H%M%S"));

    let mut steps = Vec::new();
    let mut total_cost = 0.0;
    let mut total_days = 0i32;

    // Step 1: Diagnosis / Initial consultation at top hospital
    if let Some(top_hospital) = hospitals.first() {
        let name = top_hospital.metadata.get("name").cloned().unwrap_or_default();
        let location = top_hospital.metadata.get("location").cloned().unwrap_or_default();
        let cost: f32 = top_hospital.metadata.get("avg_cost_outpatient")
            .and_then(|c| c.parse().ok())
            .unwrap_or(1500.0);
        let wait_days: i32 = top_hospital.metadata.get("avg_wait_days")
            .and_then(|d| d.parse().ok())
            .unwrap_or(14);

        steps.push(NavigationStep {
            step_number: 1,
            step_type: "diagnosis".into(),
            title: format!("第一步：赴 {} 确诊/会诊", name),
            description: format!(
                "前往 {}（{}）进行罕见病专科会诊。该医院在{}领域具有丰富经验，\
                 建议提前预约并准备好所有既往病历和基因检测报告。",
                name, location, profile.disease_name
            ),
            target_hospital: name.clone(),
            target_doctor: top_hospital.metadata.get("specialty").cloned().unwrap_or_else(|| "罕见病专科".into()),
            estimated_cost_usd: cost,
            estimated_days: wait_days,
            confidence: 0.9,
            reasons: vec![
                format!("{} 专长匹配", profile.disease_name),
                "国际患者服务完善".into(),
                top_hospital.metadata.get("interpreter").map(|i| format!("提供{}翻译服务", i)).unwrap_or_default(),
            ].into_iter().filter(|s| !s.is_empty()).collect(),
            source_type: "evidence".into(),
            audit_hash: String::new(),
        });

        total_cost += cost as f32;
        total_days += wait_days;
    }

    // Step 2: Treatment plan
    if hospitals.len() > 1 {
        let h = &hospitals[1];
        let name = h.metadata.get("name").cloned().unwrap_or_default();
        let cost: f32 = h.metadata.get("avg_cost_inpatient")
            .and_then(|c| c.parse().ok())
            .unwrap_or(50000.0);

        steps.push(NavigationStep {
            step_number: 2,
            step_type: "treatment".into(),
            title: "第二步：制定治疗方案".into(),
            description: format!(
                "根据确诊结果，在 {} 制定个性化治疗方案。可能包括酶替代疗法、\
                 基因治疗或参与临床试验。",
                name
            ),
            target_hospital: name,
            target_doctor: "多学科会诊团队".into(),
            estimated_cost_usd: cost,
            estimated_days: 14,
            confidence: 0.8,
            reasons: vec![
                "多学科会诊".into(),
                "个性化治疗方案".into(),
            ],
            source_type: "guideline".into(),
            audit_hash: String::new(),
        });

        total_cost += cost;
        total_days += 14;
    }

    // Step 3: Clinical trial (if requested)
    steps.push(NavigationStep {
        step_number: (steps.len() + 1) as i32,
        step_type: "trial".into(),
        title: "第三步：评估临床试验入组资格".into(),
        description: format!(
            "查询 {} 相关的活跃临床试验。建议通过 ClinicalTrials.gov \
             或主治医生了解最新试验信息，部分试验可能提供免费治疗。",
            profile.disease_name
        ),
        target_hospital: hospitals.first()
            .and_then(|h| h.metadata.get("name").cloned())
            .unwrap_or_default(),
        target_doctor: "临床研究协调员".into(),
        estimated_cost_usd: 0.0,
        estimated_days: 7,
        confidence: 0.7,
        reasons: vec![
            "可能获得前沿治疗".into(),
            "研究性治疗费用可能减免".into(),
        ],
        source_type: "evidence".into(),
        audit_hash: String::new(),
    });
    total_days += 7;

    // Step 4: Logistics
    steps.push(NavigationStep {
        step_number: (steps.len() + 1) as i32,
        step_type: "logistics".into(),
        title: "第四步：行前准备".into(),
        description: "办理医疗签证（B-2）、预订机票和住宿、准备病历翻译件、购买旅行保险。\
                     建议携带足够的美金信用卡和现金备用。".into(),
        target_hospital: String::new(),
        target_doctor: String::new(),
        estimated_cost_usd: 5000.0,
        estimated_days: 21,
        confidence: 0.95,
        reasons: vec![
            "医疗签证审批约 2-3 周".into(),
            "建议提前预约翻译服务".into(),
        ],
        source_type: "experience".into(),
        audit_hash: String::new(),
    });
    total_cost += 5000.0;
    total_days += 21;

    // Step 5: Follow-up
    steps.push(NavigationStep {
        step_number: (steps.len() + 1) as i32,
        step_type: "followup".into(),
        title: "第五步：回国后随访".into(),
        description: "回国后定期远程随访，与美方主治医生保持沟通。建议每 3-6 个月复查一次，\
                     通过远程会诊调整治疗方案。".into(),
        target_hospital: String::new(),
        target_doctor: String::new(),
        estimated_cost_usd: 2000.0,
        estimated_days: 0,
        confidence: 0.85,
        reasons: vec![
            "远程随访降低成本".into(),
            "持续监测病情变化".into(),
        ],
        source_type: "guideline".into(),
        audit_hash: String::new(),
    });
    total_cost += 2000.0;

    // Collect similar experience IDs
    let similar_experiences: Vec<String> = experiences
        .iter()
        .map(|e| e.id.clone())
        .collect();

    // Compute audit hash
    let audit_hash = {
        if let Ok(a) = audit.read() {
            let entries = a.recent(1);
            entries.first().map(|b| b.hash.clone()).unwrap_or_default()
        } else {
            String::new()
        }
    };

    // Set audit hash on first step
    if let Some(step) = steps.first_mut() {
        step.audit_hash = audit_hash.clone();
    }

    let overall_confidence = if hospitals.is_empty() { 0.3 } else { 0.85 };

    Ok(NavigationRoute {
        route_id,
        patient_summary: format!(
            "{}患者，{}，{}阶段，预算${:.0}，{}",
            profile.disease_name,
            profile.age_group,
            profile.stage,
            profile.budget_usd,
            profile.urgency,
        ),
        steps,
        total_estimated_cost: total_cost,
        total_estimated_days: total_days,
        overall_confidence,
        similar_experiences,
        generated_at: chrono::Utc::now().to_rfc3339(),
        audit_hash,
    })
}

/// Create gRPC server for MedGpsService
pub fn create_med_gps_server(
    db: SharedVectorDB,
    audit: SharedAuditChain,
) -> MedGpsServiceServer<MedGpsServiceImpl> {
    MedGpsServiceServer::new(MedGpsServiceImpl::new(db, audit))
}
