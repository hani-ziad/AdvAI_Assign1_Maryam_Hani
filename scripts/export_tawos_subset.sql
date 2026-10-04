-- Current TAWOS export for sprint-risk prediction.
--
-- IMPORTANT DIFFERENCE FROM THE OLD EXPORT:
--   * export ALL issues in the selected projects, including issues whose
--     current Sprint_ID is NULL; an issue may have belonged to a sprint in
--     the past and later been removed.
--   * export Issue.ID and Creation_Date.
--   * export Sprint.JiraID and Activated_Date.
--   * export Change_Log events for Sprint and Story Point fields so the
--     Python pipeline can reconstruct scope as it existed at sprint start.
--
-- Run these SELECTs in MySQL Workbench and export each result with the exact
-- filenames shown below, or use the mysql CLI in batch mode.

-- Selected projects (same eight used in the study):
--   Apache Mesos
--   Appcelerator Studio
--   Atlassian Confluence Cloud
--   Atlassian Confluence Server
--   Lsstcorp Data management
--   Mule
--   The Titanium SDK
--   Titanium Mobile Platform

-- 1) Save as data/raw/tawos/tawos_sprint.csv
SELECT
    s.ID,
    s.JiraID,
    s.Name,
    s.State,
    s.Start_Date,
    s.Activated_Date,
    s.End_Date,
    s.Complete_Date,
    TRIM(p.Name) AS Project_Name
FROM Sprint s
JOIN Project p ON p.ID = s.Project_ID
WHERE TRIM(p.Name) IN (
    'Apache Mesos',
    'Appcelerator Studio',
    'Atlassian Confluence Cloud',
    'Atlassian Confluence Server',
    'Lsstcorp Data management',
    'Mule',
    'The Titanium SDK',
    'Titanium Mobile Platform'
)
ORDER BY TRIM(p.Name), s.Start_Date, s.ID;

-- 2) Save as data/raw/tawos/tawos_issue.csv
-- Do NOT add "Sprint_ID IS NOT NULL": issues removed from a sprint after
-- planning are exactly the rows needed to reconstruct planning-time scope.
SELECT
    i.ID,
    i.Issue_Key,
    i.Type,
    i.Status,
    i.Story_Point,
    i.Priority,
    i.Creation_Date,
    i.Resolution_Date,
    i.Sprint_ID,
    i.Assignee_ID,
    TRIM(p.Name) AS Project_Name
FROM Issue i
JOIN Project p ON p.ID = i.Project_ID
WHERE TRIM(p.Name) IN (
    'Apache Mesos',
    'Appcelerator Studio',
    'Atlassian Confluence Cloud',
    'Atlassian Confluence Server',
    'Lsstcorp Data management',
    'Mule',
    'The Titanium SDK',
    'Titanium Mobile Platform'
)
ORDER BY i.ID;

-- 3) Save as data/raw/tawos/tawos_change_log.csv
-- We need Sprint history to reconstruct membership at sprint start and Story
-- Point history to reconstruct committed points at the same timestamp.
SELECT
    c.ID,
    c.Issue_ID,
    c.Field,
    c.From_Value,
    c.To_Value,
    c.From_String,
    c.To_String,
    c.Change_Type,
    c.Creation_Date,
    TRIM(p.Name) AS Project_Name
FROM Change_Log c
JOIN Issue i ON i.ID = c.Issue_ID
JOIN Project p ON p.ID = i.Project_ID
WHERE TRIM(p.Name) IN (
    'Apache Mesos',
    'Appcelerator Studio',
    'Atlassian Confluence Cloud',
    'Atlassian Confluence Server',
    'Lsstcorp Data management',
    'Mule',
    'The Titanium SDK',
    'Titanium Mobile Platform'
)
AND (
    LOWER(c.Field) LIKE '%sprint%'
    OR LOWER(c.Field) LIKE '%story%point%'
)
ORDER BY c.Issue_ID, c.Creation_Date, c.ID;
