package smile

import org.apache.commons.csv.CSVFormat
import smile.data.formula.Formula
import smile.io.Read
import smile.validation.CrossValidation
import smile.classification.RandomForest

fun main() {
    // Загрузка датасета Breast Cancer
    val csvFormat = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).setDelimiter(',').build()

    val cancerData = Read.csv(
        "C:\\Users\\aseme\\IdeaProjects1\\kotlin-for-data-science\\frameworks\\frameworks\\src\\main\\resources\\The_Cancer_data_1500_V2.csv", csvFormat)

    println("Исходный датасет:")
    println(cancerData)

    // Формула для целевой переменной
    val formula = Formula.lhs("Diagnosis")

    // Кросс-валидация с RandomForest для классификации
    val cvResult = CrossValidation.classification(
        10, formula, cancerData,
        { f, data -> RandomForest.fit(f, data) }  // Используем RandomForest
    )

    // Выводим результат кросс-валидации
    println("\nРезультаты кросс-валидации (Random Forest): $cvResult")
}
