package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// FolderInfo 存储文件夹信息
type FolderInfo struct {
	path     string
	name     string
	size     int64
	fileCount int
}

// 转换字节数为易读格式
func formatSize(bytes int64) string {
	const (
		_  = iota
		KB = 1 << (10 * iota)
		MB
		GB
		TB
	)

	switch {
	case bytes >= TB:
		return fmt.Sprintf("%.2f TB", float64(bytes)/TB)
	case bytes >= GB:
		return fmt.Sprintf("%.2f GB", float64(bytes)/GB)
	case bytes >= MB:
		return fmt.Sprintf("%.2f MB", float64(bytes)/MB)
	case bytes >= KB:
		return fmt.Sprintf("%.2f KB", float64(bytes)/KB)
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

// 递归计算文件夹大小
func calculateFolderSize(path string, wg *sync.WaitGroup, resultChan chan<- FolderInfo, errorChan chan<- error) {
	defer wg.Done()

	var totalSize int64
	var fileCount int

	// 遍历文件夹
	err := filepath.Walk(path, func(filePath string, info os.FileInfo, err error) error {
		if err != nil {
			// 处理访问权限问题
			if os.IsPermission(err) {
				return filepath.SkipDir
			}
			return err
		}

		if !info.IsDir() {
			totalSize += info.Size()
			fileCount++
		}
		return nil
	})

	if err != nil {
		errorChan <- fmt.Errorf("处理文件夹 %s 时出错: %v", path, err)
		return
	}

	resultChan <- FolderInfo{
		path:     path,
		name:     filepath.Base(path),
		size:     totalSize,
		fileCount: fileCount,
	}
}

// 获取指定目录下的所有直接子文件夹
func getSubfolders(root string) ([]string, error) {
	var folders []string

	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}

	for _, entry := range entries {
		if entry.IsDir() {
			fullPath := filepath.Join(root, entry.Name())
			folders = append(folders, fullPath)
		}
	}

	return folders, nil
}

func main() {
	// 解析命令行参数
	folderPath := flag.String("path", ".", "要分析的文件夹路径")
	ascending := flag.Bool("asc", false, "按升序排序（默认按降序）")
	details := flag.Bool("details", false, "显示详细信息")
	flag.Parse()

	// 验证文件夹是否存在
	absPath, err := filepath.Abs(*folderPath)
	if err != nil {
		fmt.Printf("无效的路径: %v\n", err)
		os.Exit(1)
	}

	info, err := os.Stat(absPath)
	if err != nil || !info.IsDir() {
		fmt.Printf("路径不存在或不是文件夹: %s\n", absPath)
		os.Exit(1)
	}

	// 开始计时
	startTime := time.Now()
	fmt.Printf("开始分析文件夹: %s\n", absPath)
	fmt.Printf("开始时间: %s\n", startTime.Format("2006-01-02 15:04:05"))
	fmt.Println("正在计算文件夹大小，请稍候...")

	// 获取所有子文件夹
	subfolders, err := getSubfolders(absPath)
	if err != nil {
		fmt.Printf("获取子文件夹失败: %v\n", err)
		os.Exit(1)
	}

	if len(subfolders) == 0 {
		fmt.Printf("在 %s 下未找到子文件夹\n", absPath)
		os.Exit(0)
	}

	// 准备并发计算
	var wg sync.WaitGroup
	resultChan := make(chan FolderInfo, len(subfolders))
	errorChan := make(chan error, len(subfolders))

	// 为每个子文件夹启动一个goroutine
	for _, folder := range subfolders {
		wg.Add(1)
		go calculateFolderSize(folder, &wg, resultChan, errorChan)
	}

	// 等待所有goroutine完成并关闭通道
	go func() {
		wg.Wait()
		close(resultChan)
		close(errorChan)
	}()

	// 收集结果
	var folderInfos []FolderInfo
	var totalSize int64
	var totalFiles int

	for info := range resultChan {
		folderInfos = append(folderInfos, info)
		totalSize += info.size
		totalFiles += info.fileCount
	}

	// 处理错误
	for err := range errorChan {
		fmt.Printf("警告: %v\n", err)
	}

	// 按大小排序
	sort.Slice(folderInfos, func(i, j int) bool {
		if *ascending {
			return folderInfos[i].size < folderInfos[j].size
		}
		return folderInfos[i].size > folderInfos[j].size
	})

	// 显示结果
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Printf("分析结果 - 文件夹: %s\n", absPath)
	fmt.Printf("总子文件夹数: %d\n", len(folderInfos))
	fmt.Printf("总文件数: %d\n", totalFiles)
	fmt.Printf("总大小: %s\n", formatSize(totalSize))
	fmt.Println(strings.Repeat("=", 80) + "\n")

	// 输出表头
	fmt.Printf("%-5s %-40s %-15s %-10s\n", "序号", "文件夹名称", "大小", "文件数")
	fmt.Println(strings.Repeat("-", 80))

	// 输出每个文件夹信息
	for i, info := range folderInfos {
		name := info.name
		if len(name) > 37 {
			name = name[:37] + "..."
		}
		fmt.Printf("%-5d %-40s %-15s %-10d\n", i+1, name, formatSize(info.size), info.fileCount)

		if *details {
			fmt.Printf("   路径: %s\n\n", info.path)
		}
	}

	// 输出完成信息
	endTime := time.Now()
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Printf("完成时间: %s\n", endTime.Format("2006-01-02 15:04:05"))
	fmt.Printf("耗时: %v\n", endTime.Sub(startTime))
}
